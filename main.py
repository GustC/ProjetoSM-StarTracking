import cv2 as cv
import numpy as np
import imutils
from os import system,name
from configs import *

def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 


def showInput(message,messageEmpty="Digite novamente",required=True,defaultValue="  ",validator = None):
  print(message)
  inputValue = defaultValue 
  while(True):
    inputValue = str(input())
    if(required and len(inputValue) > 0):
      print(messageEmpty)      
    clear()
    if(validator == None):
      break
    elif(validator(inputValue)):
      break
  return inputValue

def checkStars(value="50"):
  response = False
  if(len(value)>0):
    if(int(value) < 1):
      print("Valor deve ser maior que 0")
    elif(int(value) > 100):
      print("O valor maximo e igual a 100")
    else:
      response = True
  return response

fileName = showInput(
  message="Digite o nome do arquivo de video",
  messageEmpty="Digite o nome do arquivo de video",
)

qtyStars = showInput(
  message="Digite a quantidade de estrelas que serão rastreadas (min. 1 | max. 100) (default 50)",
  required=False,
  validator=checkStars,
  defaultValue="50"
)
if(len(qtyStars)==0):
  qtyStars = 50
else:
  qtyStars = int(qtyStars)
print(qtyStars)
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


# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
old_blur = cv.bilateralFilter(old_gray,5,10,2.5)
ret,old_threshold = cv.threshold(old_blur,minThreshold,maxThreshold,cv.THRESH_BINARY) 
# old_threshold = cv.adaptiveThreshold(old_blur,maxThreshold,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2) 
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
old_threshold = cv.dilate(old_threshold, kernel, iterations=1)
p0 = cv.goodFeaturesToTrack(old_threshold, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Read until video is completed
firstFrame = None
frameCount = 0
l = True
maxSkip = 0
cv.namedWindow("Video", cv.WINDOW_NORMAL) 
cv.namedWindow("VideoTresh", cv.WINDOW_NORMAL) 
cv.resizeWindow('Video', 600,400)
cv.resizeWindow('VideoTresh', 600,400)
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read() 

  if ret == True:
    frame_gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # blur = cv.medianBlur(frame_gray,5)
    # blur = cv.GaussianBlur(frame_gray,(5,5),0)
    new_blur = cv.bilateralFilter(frame_gray,5,10,2.5)
    ret,new_threshold = cv.threshold(new_blur,minThreshold,maxThreshold,cv.THRESH_BINARY)  
    # new_threshold = cv.adaptiveThreshold(new_blur,maxThreshold,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv.THRESH_BINARY,11,2)  
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
    cv.imshow("Video",img)
    cv.imshow("VideoTresh",new_threshold)
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):      
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
print(maxSkip)
# Closes all the frames
cv.destroyAllWindows()

