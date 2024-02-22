import cv2
from img_processing import  process_img
video_stream = cv2.VideoCapture("prueba.MTS")

ok, img = video_stream.read()
while ok:
    process_img(img)
    



    ok,img = video_stream.read()