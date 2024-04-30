import numpy as np
import cv2
import os
path = "C:/Users/Diego/Documents/Código/TFG2024/resources/videos/trim.mp4"
salida = "C:/Users/Diego/Documents/Código/TFG2024/resources/frames/trim_mp4/"
video_capture = cv2.VideoCapture(path)
i = 0
ok = True
while ok:
    ok, image = video_capture.read()
    if ok:
        print(cv2.imwrite(os.path.join(salida,f"{i}.jpg") ,img=image))
        i = i +1
    
