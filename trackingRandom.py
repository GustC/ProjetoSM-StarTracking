import cv2 as cv
import numpy as np
from configs import *

qtyStars = 50
fileName = "video_estrelas_noite.mp4"
# video_estrelas_noite.mp4
cap = cv.VideoCapture(fileName)
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# params for ShiTomasi corner detection
# minDistance é a menor distancia entre os melhores pontos
# qualityLevel é a qualidade aceitavel para cada ponto
# max Corners é a quantidade max. de pontos 
config = ConfigApp(typeDay=typeDayNight,qtyStars=qtyStars)

feature_params = config.configFeature
# Parameters for lucas kanade optical flow
lk_params = config.configLk
# c.configFeature
minWidthDif = 100
minHeightDif = 20

minThreshold = 100
maxThreshold = 255

framesToClear = 25
kernel = np.ones((5,5), np.uint8)

#trackbar definitions
trackbar_type = 'Speed:'
trackbar_value = 'Value'
window_name = 'Video'
max_value = 100
default_speed = 100

def video_speed_demo(val):
    global default_speed 
    default_speed = cv.getTrackbarPos(trackbar_type, window_name)
    if default_speed == 0:
        default_speed = 1
    cv.waitKey(default_speed)



# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
old_blur = cv.bilateralFilter(old_gray,5,10,2.5)
ret,old_threshold = cv.threshold(old_blur,minThreshold,maxThreshold,cv.THRESH_BINARY) 
old_threshold = cv.dilate(old_threshold, kernel, iterations=1)
p0 = cv.goodFeaturesToTrack(old_threshold, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Read until video is completed
firstFrame = None
frameCount = 0
l = True
maxSkip = 0
cv.namedWindow(window_name, cv.WINDOW_NORMAL) 
cv.resizeWindow(window_name, 600,400)
# uv = utilsVideo(cap)
cv.createTrackbar(trackbar_type, window_name, default_speed, max_value, video_speed_demo)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read() 

    if ret == True:
        frame_gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        new_blur = cv.bilateralFilter(frame_gray,5,10,2.5)
        ret,new_threshold = cv.threshold(new_blur,minThreshold,maxThreshold, cv.THRESH_BINARY) 
        new_threshold = cv.dilate(new_threshold, kernel, iterations=1)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_threshold, new_threshold, p0, None, **lk_params)    
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        framesSkips = 0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        

        frameCount += 1
        if(frameCount == framesToClear):
            frameCount = 0    
            old_threshold = new_threshold
            p0 = cv.goodFeaturesToTrack(old_threshold, mask = None, **feature_params)
            mask = np.zeros_like(frame)
            img = cv.add(frame,mask)
        else:
            img = cv.add(frame,mask) 
        # if(maxSkip < framesSkips):
        #   maxSkip = framesSkips
        cv.imshow(window_name,img)
        # Press Q on keyboard to  exit
        if cv.waitKey(default_speed) & 0xFF == ord('q'):      
            break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv.destroyAllWindows()