from ultralytics import YOLO

"""
    Convierte de modelo pytorch a modelo openvino, para gráficas Intel integradas
"""

model = YOLO('C:/Users/Diego/Documents/Codigo/TFG2024/runs/detect/trainOriginal/weights/best.pt')
model.export(format='openvino')