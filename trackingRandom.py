
import cv2 as cv
import numpy as np
from tkinter import *
from configs import *
import queue
import threading

class TrackingRandom:
    def __init__(self,root,videoCapture):
        self.cap = videoCapture
        self.windowName = "Modo Random"
        self.window = Toplevel(root)
        self.window.title = "Controles"
        self.btnPause = Button(self.window,text="Pause",command=self.pause,bg="red",fg="white",width=20)
        self.btnPlay = Button(self.window,text="Play",command=self.play,bg="green",fg="white",width=20)
        self.btnStop = Button(self.window,text="Stop",command=self.stop,bg="blue",fg="white",width=20)
        self.btnPause.grid(row=0,column=0,stick=NSEW)
        self.btnPlay.grid(row=0,column=1,stick=NSEW)
        self.btnStop.grid(row=0,column=2,stick=NSEW)
        self.isPause = False
        self.isStop = False
        self.config = ConfigApp(typeDay=typeDayNight,qtyStars=50)
        self.feature_params = self.config.configFeature
        self.lk_params = self.config.configLk
        self.framesToClear = 25
        self.max_value = 100
        self.default_speed = 100
        self.minThreshold = 100
        self.maxThreshold = 255

    def pauseVideo(self): 
        self.isPause = True
        return
    def pause(self):
        pauseVid = threading.Thread(target=self.pauseVideo)
        pauseVid.start()
        return  
    def playVideo(self):
        self.isPause = False
        return
    def play(self):
        playVid = threading.Thread(target=self.playVideo)
        playVid.start()
        return
    def stopVideo(self):
        self.isStop = True
        return
    def stop(self):
        stopVid = threading.Thread(target=self.stopVideo)
        stopVid.start()

    def startVideo(self):
        self.window.lift()
        ret, old_frame = self.cap.read()
        kernel = np.ones((5,5), np.uint8)

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        old_blur = cv.bilateralFilter(old_gray,5,10,2.5)
        ret,old_threshold = cv.threshold(old_blur,self.minThreshold,self.maxThreshold,cv.THRESH_BINARY) 
        old_threshold = cv.dilate(old_threshold, kernel, iterations=1)
        p0 = cv.goodFeaturesToTrack(old_threshold, mask = None, **self.feature_params)

        mask = np.zeros_like(old_frame)

        color = np.random.randint(0,255,(100,3))

        def video_speed_demo(val):
            self.default_speed = cv.getTrackbarPos("Speed:", self.windowName)
            if self.default_speed == 0:
                self.default_speed = 1            
            cv.waitKey(self.default_speed)

        frameCount = 0
        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL) 
        cv.resizeWindow(self.windowName, 600,400)
        cv.createTrackbar("Speed:", self.windowName, self.default_speed, self.max_value, video_speed_demo)

        while(self.cap.isOpened()):
            while(self.isPause):
                print("pausado")
            if(self.isStop):
                break
            # Capture frame-by-frame
            ret, frame = self.cap.read() 

            if ret == True:
                frame_gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
                new_blur = cv.bilateralFilter(frame_gray,5,10,2.5)
                ret,new_threshold = cv.threshold(new_blur,self.minThreshold,self.maxThreshold, cv.THRESH_BINARY) 
                new_threshold = cv.dilate(new_threshold, kernel, iterations=1)
                # calculate optical flow
                p1, st, err = cv.calcOpticalFlowPyrLK(old_threshold, new_threshold, p0, None, **self.lk_params)    
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
                if(frameCount == self.framesToClear):
                    frameCount = 0    
                    old_threshold = new_threshold
                    p0 = cv.goodFeaturesToTrack(old_threshold, mask = None, **self.feature_params)
                    mask = np.zeros_like(frame)
                    img = cv.add(frame,mask)
                else:
                    img = cv.add(frame,mask) 
                cv.imshow(self.windowName,img)
                # Press Q on keyboard to  exit
                
                key = cv.waitKey(self.default_speed)
                if key == ord('q'):      
                    break
                elif key == ord('p'):
                    cv.waitKey(0)

            # Break the loop
            else: 
                print("dale")
                break

        # When everything done, release the video capture object
        self.cap.release()
        cv.destroyAllWindows()
        self.window.destroy()


