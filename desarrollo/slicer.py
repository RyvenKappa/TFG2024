import cv2
import numpy as np

video_stream = cv2.VideoCapture("prueba.MTS")
ok, img = video_stream.read()
img = img[0:1080,0:1550]
frame1 = None
frame2 = None
while ok:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.convertScaleAbs(gray,1.5,1)
    
    
    ret,thresh = cv2.threshold(gray,50,255,0)
    thresh = cv2.bitwise_not(thresh)
    c,hierarchy = cv2.findContours(thresh, 1, 2)
    contours = sorted(c, key=cv2.contourArea,reverse=True)[:10]

    
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>115000 and area<350000:
            x,y,w,h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio>=0.6 and ratio <= 0.9:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                rectangles.append(box)
    if len(rectangles)>0:
        print(rectangles)
        img = cv2.drawContours(img,rectangles,-1,(0,0,255),2)
        rectangles.clear()

    cv2.imshow("Shapes", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ok,img = video_stream.read()
    img = img[0:1080,0:1550]
