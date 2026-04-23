import json
import time
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm


# Paths
_HERE = Path(__file__).parent
DEFAULT_WEIGHTS = _HERE / "model_weights" / "best_model.pth"
DEFAULT_CONFIG  = _HERE / "model_weights" / "model_config.json"


# Config 
FALLBACK_CONFIG = {
    "model_name": "convnext_tiny",
    "num_classes": 2,
    "class_names": ["bad", "good"],   
    "img_size": 224,
    "drop_rate": 0.2,
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# Model 
class ConvNeXtWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=0
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        dummy = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"])
        feat = self.backbone.forward_features(dummy)
        feat_dim = feat.shape[1]

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(cfg["drop_rate"] / 2),
            nn.Linear(256, cfg["num_classes"]),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x


# Load Model 
def load_model(weights_path=None, config_path=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = Path(weights_path or DEFAULT_WEIGHTS)
    config_path  = Path(config_path or DEFAULT_CONFIG)

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        print("[WARN] Using fallback config")
        cfg = FALLBACK_CONFIG.copy()

    for k, v in FALLBACK_CONFIG.items():
        cfg.setdefault(k, v)

    model = ConvNeXtWrapper(cfg)

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    new_state = {}
    for k, v in state.items():
        if not k.startswith("backbone.") and "stages" in k:
            new_state["backbone." + k] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()

    print(f"[INFO] Loaded {cfg['model_name']} on {device}")

    return model, cfg["class_names"], cfg["img_size"], device


# Transform 
def get_inference_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# Predict
def predict(model, image, class_names, img_size=224, device=None):

    if device is None:
        device = next(model.parameters()).device

    transform = get_inference_transform(img_size)

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = probs.argmax().item()
    label = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            cls: probs[i].item() for i, cls in enumerate(class_names)
        },
        "is_defect": ("bad" in label.lower()),
        "latency_ms": latency_ms,
    }


# Overlay 
def draw_prediction_overlay(frame, result, fps=0.0):

    out = frame.copy()
    h, w = out.shape[:2]

    label = result["label"]
    conf  = result["confidence"]

    is_def = result.get("is_defect", False)
    color = (0, 0, 255) if is_def else (0, 255, 0)

    cv2.rectangle(out, (0, 0), (w, 60), (30, 30, 30), -1)

    text = f"{label.upper()} {conf*100:.1f}%"
    cv2.putText(out, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    bar_w = int(w * conf)
    cv2.rectangle(out, (0, h - 10), (w, h), (50, 50, 50), -1)
    cv2.rectangle(out, (0, h - 10), (bar_w, h), color, -1)

    if fps > 0:
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return out


# Self Test 
if __name__ == "__main__":
    print("Running self-test...\n")
    model, cls, sz, dev = load_model()
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    res = predict(model, dummy, cls, sz, dev)
    print("Prediction:", res)