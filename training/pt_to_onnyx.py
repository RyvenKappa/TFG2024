from ultralytics import YOLO

"""
    Convierte de modelo pytorch a modelo onnx
"""

model = YOLO("src/models/obb/best.pt")
model.export(format='onnx')