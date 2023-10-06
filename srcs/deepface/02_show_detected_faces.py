# deepface의 face_detector 와 retinaface의 extract_faces를 사용하여 얼굴을 인식하는 코드
# 정확도는 retinaface가 더 높음
# 일반적인 deepface.detectFace() 는 하나의 얼굴만 인식함

from deepface import DeepFace
from deepface.detectors import FaceDetector
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2

### set image path
img1 = "/Users/chanwoo4267/Downloads/deepface/crowd_images/1.jpg"

### set detector
detector_name = "opencv"

img = cv2.imread(img1)
detector = FaceDetector.build_model(detector_name)
obj = FaceDetector.detect_face(detector, detector_name, img)

print("there's ", len(obj), " faces in this image")

retina_faces = RetinaFace.extract_faces(img1)

for retina_face in retina_faces:
    plt.imshow(retina_face)
    plt.axis('off')
    plt.show()

print("there's ", len(retina_faces), " faces in this image")