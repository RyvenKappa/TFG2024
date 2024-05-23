import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
cap = cv.VideoCapture("resources/videos/P9_10.mp4")# In place of zero we gonna use path of the video file.
 
    
ddepth = cv.CV_16S
kernel_size = 3
blur_list = []
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

        # if frame is read correctly ret is True
    if ret:
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame

        dst = cv.Laplacian(gray,ddepth,ksize=kernel_size)
        blur_list.append(dst.var())
        abs_dst = cv.convertScaleAbs(dst)
        cv.imshow("cosa",abs_dst)
    else:
        break

print("hola")    
# When everything done, release the capture
cap.release()

plt.plot(np.diff(blur_list))
plt.show()