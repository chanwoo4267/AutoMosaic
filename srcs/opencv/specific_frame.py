import numpy as np
import cv2

cap = cv2.VideoCapture("/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4")

amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frames of video

frame_num = 200 # frame number to extract

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

res, frame = cap.read()

cv2.imshow('result', frame)

while True:
    ch = 0xFF & cv2.waitKey(1) # Wait for a second
    if ch == 27:
        break