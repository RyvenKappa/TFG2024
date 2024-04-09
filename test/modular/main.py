import cv2
import numpy as np
import filter_bank

path_video = "test.mp4"
#Abrimos el video
video_capture = cv2.VideoCapture(path_video)
ok, img = video_capture.read()
bank = filter_bank.filters()
while ok:
    frame = bank.proccess_frame(img)

    cv2.imshow("Shapes", img)
    k = cv2.waitKey(0)
    ok, img = video_capture.read()



