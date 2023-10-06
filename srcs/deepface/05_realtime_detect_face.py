import numpy as np
import cv2
from retinaface import RetinaFace

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

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
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 10)

    cv2.imshow('result', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
