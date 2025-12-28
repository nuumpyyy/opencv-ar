import cv2 as cv
from threading import Thread

# Any instance of the Webcam class is initialized with the current frame detected by the camera, that is, a single image
# Creating a new instance will always initiate video capture
class Webcam:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_FPS, 20)
        self.current_frame = self.cap.read()[1]
        #cv.imshow("Current frame", self.current_frame)
        #cv.waitKey(3000)

    # create thread for capturing images
    def start(self):
        Thread(target=self.update_frame, args=()).start()

    def update_frame(self):
         while True:
            self.current_frame = self.cap.read()[1]

    # get current frame
    def get_current_frame(self):
        return self.current_frame
        
webcam = Webcam()
cv.imshow("Current frame", webcam.get_current_frame())
cv.waitKey(3000)