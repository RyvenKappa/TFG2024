import cv2
import numpy as np

# initialize variables updated by function
selected_point = False
point = ()
old_points = ([[]])



# define function to manually select object to track
def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    # record coordinates of mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)


# associate select function with window Selector
cv2.namedWindow('Optical Flow')
cv2.setMouseCallback('Optical Flow', select_point)


cap = cv2.VideoCapture("Trim2.MTS")
#First Frame

ok, frame = cap.read()

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use Shi-Tomasi to detect object corners / edges from initial frame
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))
# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(100, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break
    # covert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        # update object corners by comparing with found edges in initial frame
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points, None,
                                                         **parameter_lucas_kanade)

        # overwrite initial frame with current before restarting the loop
        frame_gray_init = frame_gray.copy()
        # update to new edges before restarting the loop
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        # draw line between old and new corner point with random colour
        canvas = cv2.line(canvas, (int(x), int(y)), (int(j), int(k)), (0, 255, 0), 3)
        # draw circle around new position
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    result = cv2.add(frame, canvas)
    cv2.imshow('Optical Flow', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break