from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.to('cuda')
results = model.train(data='C:/Users/Diego/Documents/TFG2024/training/sp.yaml',epochs=100,imgsz=640,device='0',batch=8)