import numpy as np
import cv2 as cv
cap = cv.VideoCapture("C:/Users/Diego/Documents/Código/TFG2024/resources/videos/Trim2.MTS")# In place of zero we gonna use path of the video file.
 
    

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

        # if frame is read correctly ret is True
    if ret:
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame
        cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
    else:
        print("camer not streaming")
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()