from ultralytics import YOLO

model = YOLO("C:/Users/Diego/Documents/TFG2024/runs/detect/train/weights/best.pt")
model.to("cuda")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="test2.mp4",save=True,save_crop=True)