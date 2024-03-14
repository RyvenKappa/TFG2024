import cv2
import numpy as np
from torch import cuda
from ultralytics import YOLO
model = YOLO("yolov8n.pt") 
model.info()
model.to('cuda')
results = model.predict(source="test.mp4",show=True)
from ultralytics import SAM

# # Load a model
# model = SAM('sam_b.pt')
# model.to('cuda')

# # Display model information (optional)
# model.info()

# model.predict(source="test.mp4",show=True)

from ultralytics import FastSAM

# Load a model
model = FastSAM('FastSAM-s.pt')
model.to('cuda')

# Display model information (optional)
model.info()

model.predict(source="test.mp4",show=True)