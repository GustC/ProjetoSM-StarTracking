import cv2 as cv
import numpy as np

cap = cv.VideoCapture("video_estrelas_noite.mp4")

# create old frame
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#trackbar definitions
trackbar_type = 'Speed:'
trackbar_value = 'Value'
window_name = 'Frame'
max_value = 100
default_speed = 100

def video_speed_demo(val):
    global default_speed 
    default_speed = cv.getTrackbarPos(trackbar_type, window_name)
    if default_speed == 0:
        default_speed = 1
    cv.waitKey(default_speed)


mask = np.zeros_like(frame)

# Lucas Kanade params
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# funcoes do mouse
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points,frame,mask
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        mask = np.zeros_like(frame)
        frame = cv.add(frame,mask)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


cv.namedWindow(window_name,cv.WINDOW_NORMAL)
cv.resizeWindow(window_name,1200,900)
cv.setMouseCallback(window_name, select_point)

point_selected = False
point = ()
oldpoints = np.array([[]])

cv.createTrackbar(trackbar_type, window_name, default_speed, max_value, video_speed_demo)

while True:
    _, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if point_selected is True:
        #cv.circle(frame, point, 5, (0, 0, 255), 2)
        
        new_points, status, error = cv.calcOpticalFlowPyrLK(
            old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x,y = new_points.ravel()
        c,d = old_points.ravel()
        mask = cv.line(mask, (x,y),(c,d), (0,255,0), 3)
        cv.circle(frame, (x,y), 5, (0,255,0), -1)
        frame = cv.add(frame,mask) 

    cv.imshow("Frame", frame)
    #cv.imshow("First level", cv.pyrDown(frame))

    if cv.waitKey(default_speed) and 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()