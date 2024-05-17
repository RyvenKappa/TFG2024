from ultralytics import YOLO

"""
    Convierte de modelo pytorch a modelo openvino, para gráficas Intel integradas
"""

model = YOLO('runs/obb/train10/weights/best.pt')
model.export(format='openvino')