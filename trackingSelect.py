import cv2 as cv
import numpy as np
from tkinter import *
import threading


class TrackingSelect:
    def __init__(self,root,videoCapture):   
        self.cap = videoCapture
        self.windowName = "Modo Selecionavel"
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

        self.lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))   
        self.point_selected = False
        self.point = ()
        self.old_points = np.array([[]])
        
        self.max_value = 100
        self.default_speed = 100

        _,self.frame = self.cap.read()

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
        old_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(self.frame)

        def video_speed_demo(val):
            self.default_speed = cv.getTrackbarPos("Speed:", self.windowName)
            if self.default_speed == 0:
                self.default_speed = 1            
            cv.waitKey(self.default_speed)
        def select_point(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                self.point = (x, y)
                self.mask = np.zeros_like(self.frame)
                self.frame = cv.add(self.frame,self.mask)
                self.point_selected = True
                self.old_points = np.array([[x, y]], dtype=np.float32)

        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL) 
        cv.resizeWindow(self.windowName, 800,600)
        cv.setMouseCallback(self.windowName, select_point)
        cv.createTrackbar("Speed:", self.windowName, self.default_speed, self.max_value, video_speed_demo)        

        while True:
            while(self.isPause):
                print("pausado")
            if(self.isStop):
                break
            _, self.frame = self.cap.read()
            gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

            if self.point_selected is True:
                
                new_points, status, error = cv.calcOpticalFlowPyrLK(
                    old_gray, gray_frame, self.old_points, None, **self.lk_params)
                old_gray = gray_frame.copy()
                self.old_points = new_points

                x,y = new_points.ravel()
                c,d = self.old_points.ravel()
                self.mask = cv.line(self.mask, (x,y),(c,d), (0,255,0), 3)
                cv.circle(self.frame, (x,y), 5, (0,255,0), -1)
                self.frame = cv.add(self.frame,self.mask) 

            cv.imshow(self.windowName, self.frame)

            key = cv.waitKey(self.default_speed)
            if key == ord('q'):      
                break
            elif key == ord('p'):
                cv.waitKey(0)
            elif key == 27:
                break
     
        self.cap.release()
        cv.destroyAllWindows()
        self.window.destroy()