# Description: test for detecting given face, then mosaic except given face
# each frame, it prints distance between given face and detected face
# to continue, press esc for next frame

from deepface import DeepFace
import numpy as np
import cv2
from retinaface import RetinaFace

cap = cv2.VideoCapture("/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4")
v = 50

# target image
img1_path = "/Users/chanwoo4267/Downloads/deepface/sample_video/2.png"

img1_faces = RetinaFace.extract_faces(img1_path)
for img1_face in img1_faces:
    img1 = img1_face

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    faces = RetinaFace.detect_faces(frame)
    for i in range(len(faces)):

        temp_str = 'face_' + str(i+1)
        startX = faces[temp_str]['facial_area'][0]
        startY = faces[temp_str]['facial_area'][1]
        endX = faces[temp_str]['facial_area'][2]
        endY = faces[temp_str]['facial_area'][3]

        obj = DeepFace.verify(img1_path = img1, img2_path = frame[startY:endY, startX:endX], model_name = "VGG-Face", enforce_detection = False)
        if (obj["distance"] > 0.3) :
            roi_f = frame[startY:endY, startX:endX]
            roi = cv2.resize(roi_f, (roi_f.shape[1] // v, roi_f.shape[0] // v))
            roi = cv2.resize(roi, (roi_f.shape[1], roi_f.shape[0]), interpolation=cv2.INTER_AREA)
            frame[startY:endY, startX:endX] = roi
            print(temp_str + " : " + str(startX) + ", " + str(startY) + ", " + str(endX) + ", " + str(endY))
            print(obj["distance"])
        else :
            print("same person")
            print(temp_str + " : " + str(startX) + ", " + str(startY) + ", " + str(endX) + ", " + str(endY))
            print(obj["distance"])
            
    
    cv2.imshow('result', frame)

    while(1) :
        k = cv2.waitKey(30) & 0xff
        if k == 27: # Esc 키를 누르면 종료
            break
    

cap.release()
cv2.destroyAllWindows()
