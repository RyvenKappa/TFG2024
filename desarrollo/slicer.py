import cv2
import numpy as np

video_stream = cv2.VideoCapture("prueba.MTS")

ok, img = video_stream.read()

while ok:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)

    ret,thresh = cv2.threshold(gray,50,255,0)
    c,hierarchy = cv2.findContours(thresh, 1, 2)
    contours = sorted(c, key=cv2.contourArea,reverse=True)[:10]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area<10000:
            try:
                contours.remove(contour)
            except:
                None

    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.5 and ratio <= 1.5:
                img = cv2.drawContours(gray, [cnt], -1, (0,255,255), 3)
                cv2.putText(img,"", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Shapes", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ok,img = video_stream.read()