import cv2
import numpy as np
import Jetson.GPIO as GPIO
from time import time
 
filter = 0
window = ''
 
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=10 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
      )
     )
 
def callback_button_press(channel):
    print("hi")
    increment = lambda: filter + 1 if filter < 2 else 0
    global filter
    filter = increment()
    cv2.destroyWindow(window)
 
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
channel = 7
GPIO.setup(channel, GPIO.IN)
GPIO.add_event_detect(channel, GPIO.FALLING, callback=callback_button_press)
cap = cv2.VideoCapture(gstreamer_pipeline(),cv2.CAP_GSTREAMER)
times = []
sobel_size = 1
while True:
    ret, frame = cap.read()
    print(GPIO.input(channel))
    start = time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray_frame, (1, 1), 1) #kernel size can differ but must be positive and odd
    if filter == 0:
        sobelx = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=sobel_size)#kernel size must be 1, 3, 5 or 7
        sobel_x = cv2.normalize(sobelx, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        end = time()
        window = 'sobel x'
        cv2.imshow('sobel x', sobel_x)
    elif filter == 1:
        sobely = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=sobel_size)
        sobel_y = cv2.normalize(sobely, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        end = time()
        window = 'sobel y'
        cv2.imshow('sobel y', sobel_y)
    elif filter == 2:
        sobelx = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=sobel_size)#kernel size must be 1, 3, 5 or 7
        sobely = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=sobel_size)
        sobel_xy = cv2.sqrt(cv2.multiply(sobelx, sobelx) + cv2.multiply(sobely, sobely))
        sobel_xy = cv2.normalize(sobel_xy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        end = time()
        window = 'sobel xy'
        cv2.imshow('sobel xy', sobel_xy)
    #print("time processing: ", end - start)
    times.append(end - start)
    if len(times) == 100:
        print('AVG Time for 100 frames: ', sum(times)/100)
        times.clear()
    if cv2.waitKey(1) == ord('w'):
        increment = lambda: filter + 1 if filter < 2 else 0
        filter = increment()
        cv2.destroyWindow(window)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()