from deepface import DeepFace
import numpy as np
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace

cap = cv2.VideoCapture("/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4")

print("input time to detect face (s) : ")
given_time = int(input())

amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frames of video
fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(3))  # Width of the frames in the video
frame_height = int(cap.get(4))  # Height of the frames in the video

current_frame = given_time * fps

if current_frame >= amount_of_frames:
    print("Given time is out of video running time")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1) # must -1 because frame index starts from 0

res, frame = cap.read()

retina_faces = RetinaFace.extract_faces(frame)

idx = 100
for retina_face in retina_faces:
    plt.imshow(retina_face)
    plt.axis('off')
    plt.show()

    # save each image in files
    cv2.imwrite("/Users/chanwoo4267/Downloads/deepface/crowd_images/" + str(idx) + ".jpg", cv2.cvtColor(retina_face, cv2.COLOR_RGB2BGR)) # save image in RGB format, not BGR
    idx += 1