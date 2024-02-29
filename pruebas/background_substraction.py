import cv2 as cv


backSub = cv.createBackgroundSubtractorMOG2()
backSub2 = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture("test.mp4")
ret, frame = capture.read()
frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
fondo = frame.copy()
kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))

while True:
    ret, frame = capture.read()
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if frame is None:
        break
    fgMask = backSub2.apply(frame)

    fgMask = cv.morphologyEx(fgMask,cv.MORPH_CLOSE,kernel)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break