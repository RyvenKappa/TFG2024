import cv2 
import numpy as np
cap = cv2.VideoCapture("test.mp4")
#First Frame

ok, frame = cap.read()

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# create canvas to paint on
hsv_canvas = np.zeros_like(frame)
# set saturation value (position 2 in HSV space) to 255
hsv_canvas[..., 1] = 255


while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # compare initial frame with current frame
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    # get x and y coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # set hue of HSV canvas (position 1)
    hsv_canvas[..., 0] = angle*(180/(np.pi/2))
    # set pixel intensity value (position 3
    hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow (dense)', frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # set initial frame to current frame
    frame_gray_init = frame_gray