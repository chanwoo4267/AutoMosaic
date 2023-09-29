import numpy as np
import cv2
from matplotlib import pyplot as plt

# img_source = "/Users/chanwoo4267/Downloads/deepface/crowd_images/3.jpg"
img_source = "/Users/chanwoo4267/Downloads/deepface/opencv/self_camera/self.jpg"

image = cv2.imread(img_source)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### deepface face detector
from deepface import DeepFace
from retinaface import RetinaFace

faces = RetinaFace.detect_faces(img_source)

for i in range(len(faces)):
    temp_str = 'face_' + str(i+1)
    startX = faces[temp_str]['facial_area'][0]
    startY = faces[temp_str]['facial_area'][1]
    endX = faces[temp_str]['facial_area'][2]
    endY = faces[temp_str]['facial_area'][3]
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 10)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.xticks([]), plt.yticks([]) 
plt.show()

### openCV haarcascade

# xml = '/Users/chanwoo4267/Downloads/deepface/opencv/haarcascades_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(xml)
# faces = face_cascade.detectMultiScale(gray, 1.2, 5)

# print("Number of faces detected: " + str(len(faces)))

# if len(faces):
#     for (x,y,w,h) in faces:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),10)

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
# plt.xticks([]), plt.yticks([]) 
# plt.show()