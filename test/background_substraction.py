import cv2 as cv


backSub = cv.createBackgroundSubtractorMOG2()
backSub2 = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture("prueba.MTS")

#Añadido para guardar video de background substracting
#out = cv.VideoWriter("NoBackground.avi",int(cv.VideoWriter.fourcc(*'XVID')),25.0,(1920,1080),isColor=False)

ret, frame_or = capture.read()
frame = cv.cvtColor(frame_or,cv.COLOR_BGR2GRAY)
fondo = frame.copy()
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))

while True:
    ret, frame_or = capture.read()
    frame = cv.cvtColor(frame_or,cv.COLOR_BGR2GRAY)
    if frame is None:
        break
    fgMask = backSub2.apply(frame)
    fgMask = cv.morphologyEx(fgMask,cv.MORPH_OPEN,kernel)
    #out.write(fgMask)
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(1)
    #if cv.waitKey(1) & 0xFF == ord('q'):
    #    print("hola3")
    #    out.release()

#out.release()
capture.release()