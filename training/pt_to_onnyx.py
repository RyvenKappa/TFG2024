from ultralytics import YOLO

"""
    Convierte de modelo pytorch a modelo onnx
"""

model = YOLO('C:/Users/Diego/Documents/Código/TFG2024/runs/detect/train/weights/best.pt')
model.export(format='onnx')