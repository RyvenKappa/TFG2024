from ultralytics import YOLO

model = YOLO("C:/Users/Diego/Documents/TFG2024/runs/detect/train/weights/best.pt")
model.to("cuda")
results = model.predict(source="testCompleto.mov",show=True)