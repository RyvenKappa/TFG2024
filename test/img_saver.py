import cv2
import os
path = "test.mp4"
salida = "C:/Users/Diego/Documents/TFG2024/imagenes_prueba"
video_capture = cv2.VideoCapture(path)
i = 0
ok = True
while ok:
    ok, img = video_capture.read()
    if ok:
        print(cv2.imwrite(os.path.join(salida,f"{i}.jpg") ,img=img))
        i =i +1
    
