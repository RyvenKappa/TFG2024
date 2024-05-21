from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
model = YOLO("runs/obb/train10/weights/best_openvino_model")

#model = YOLO("runs/detect/train/weights/best.onnx")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="resources/videos/23_NT_R1_J1_P9_10.mp4",save=False)