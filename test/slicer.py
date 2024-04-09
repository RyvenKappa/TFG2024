import cv2
import numpy as np

video_stream = cv2.VideoCapture("test.mp4")
ok, img = video_stream.read()
img = img[0:1080,0:1550]
frame1 = None
frame2 = None

def frame_analisis(region):
    #Detectamos de que lado es
    print(len(region))
    print(region)
    if region[0]<1920/2:
        pass
    else:
        pass

while ok:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #gray = cv2.convertScaleAbs(gray,1.5,1)
    
    
    ret,thresh = cv2.threshold(gray,50,255,0)
    thresh = cv2.bitwise_not(thresh)
    c,hierarchy = cv2.findContours(thresh, 1, 2)
    contours = sorted(c, key=cv2.contourArea,reverse=True)[:10]

    
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>115000 and area<350000:
            x,y,w,h = cv2.boundingRect(cnt)
            cnt_len = cv2.arcLength(cnt,True)
            contorno = cv2.approxPolyDP(cnt,0.035*cnt_len,closed=True)
            ratio = float(w)/h
            if ratio>=0.6 and ratio <= 0.9:
                #rect = cv2.minAreaRect(cnt)
                #box = cv2.boxPoints(rect)
                #box = np.intp(box)
                #rectangles.append(box)
                frame_analisis(contorno[0][0])
                rectangles.append(contorno)
    cv2.drawContours(img,rectangles,-1,(0,255,0),3)
    cv2.imshow("Shapes", img)
    rectangles.clear()
    k = cv2.waitKey(0)
    ok,img = video_stream.read()
    img = img[0:1080,0:1550]
