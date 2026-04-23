Weld Defect Detection 

HAN_GOOD = GOOD / NO DEFECT
HAN_BAD = BAD / HAS DEFECT

1. Pretrained ConvNeXt tiny - finetuned with roboflow weld defect dataset containing 3000+ images
2. Inference using external images - predicts weld quality and outputs confidence score and class label
3. A small streamlit interface which takes image as file upload, and returns confidence score and class label
4. A Python GUI which allows multiple cameras to record real-time images and predicts class labels
5. Includes data logging as well
