import tkinter
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import os
import threading

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.switch_variable = tkinter.StringVar(value="random")
        self.random_button = tkinter.Radiobutton(window, text="Aleatorio", variable=self.switch_variable,
                            indicatoron=False, value="random" ) 
        self.select_button = tkinter.Radiobutton(window, text="Selecionavel", variable=self.switch_variable,
                                    indicatoron=False, value="select")

        self.random_button.grid(column=0,row=0,stick=tkinter.W,padx=10,pady=10)
        self.select_button.grid(column=1,row=0,padx=10,pady=10)

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Selecionar video", command=self.abrirVideo)
        # self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True,side=tkinter.BOTTOM)
        self.btn_snapshot.grid(row=2,stick=tkinter.W+tkinter.E,columnspan=2)
             
        # After it is called once, the update method will be automatically called every delay milliseconds

        self.window.mainloop()
    def abrirVideo(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.window.filename =  filedialog.askopenfilename(initialdir = path,title = "Escolha o arquivo",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
        if(len(self.window.filename) > 0):
            self.vid = MyVideoCapture(self.window.filename,self.switch_variable,self.window)
            self.vid.start()
 
class MyVideoCapture:
    def __init__(self, video_source=0,option="random",root=None):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.option = option
        self.root = root
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(800)
        self.height = self.vid.get(600)


    def start(self):
        if self.vid.isOpened():
            if(self.option.get() == "random"):
                from trackingRandom import TrackingRandom
                algTrack = TrackingRandom(self.root,self.vid)
                start = threading.Thread(target=algTrack.startVideo)
                start.start()
            else:
                from trackingSelect import TrackingSelect
                algTrack = TrackingSelect(self.root,self.vid)
                start = threading.Thread(target=algTrack.startVideo)
                start.start()                
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

App(tkinter.Tk(),"Start tracking")