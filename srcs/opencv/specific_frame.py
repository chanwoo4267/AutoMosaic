import numpy as np
import cv2

cap = cv2.VideoCapture("/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4")

given_time = 3 # set video to 3sec

amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frames of video
fps = cap.get(cv2.CAP_PROP_FPS) # frame rate of video

current_frame = given_time * fps

if current_frame >= amount_of_frames:
    print("Given time is out of video running time")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1) # must -1 because frame index starts from 0

res, frame = cap.read()

cv2.imshow('result', frame)

while True:
    ch = 0xFF & cv2.waitKey(1) # Wait for a second
    if ch == 27:
        break