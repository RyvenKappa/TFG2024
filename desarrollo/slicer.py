import cv2
import numpy as np

video_stream = cv2.VideoCapture("prueba.MTS")

ok, img = video_stream.read()

while ok:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    ret,thresh = cv2.threshold(gray,50,255,0)
    c,hierarchy = cv2.findContours(thresh, 1, 2)
    contours = sorted(c, key=cv2.contourArea,reverse=True)[:10]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area<100000 and area>1000000:
            try:
                contours.remove(contour)
            except:
                None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>150000 and area<1000000:
            x,y,w,h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio>=0.5 and ratio <= 1.5:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                img = cv2.drawContours(img,[box],0,(0,0,255),2)

    cv2.imshow("Shapes", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ok,img = video_stream.read()