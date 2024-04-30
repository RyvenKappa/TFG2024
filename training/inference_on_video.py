from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
model = YOLO("runs/detect/train8/weights/best.pt")

#model = YOLO("runs/detect/train/weights/best.onnx")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="resources/videos/trim.mp4",save=True)
