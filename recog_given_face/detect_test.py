# Description: This file is for testing face recognition using detect_face vs extract_face
# not completed yet

from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt

### set image path
img1_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_1.webp"
img2_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_2.jpeg"
img3_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_3.png"
img4_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_4.png"
img5_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_5.webp"
img6_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_6.jpeg"
img7_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_7.jpeg"
img8_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_8.jpeg"
img9_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_9.jpeg"
img10_path = "/Users/chanwoo4267/Downloads/deepface/recognition_images/dicaprio_10.jpeg"

### specify model
model_name = "VGG-Face"

faces = RetinaFace.detect_faces(img1_path)
for i in range(len(faces)):

    temp_str = 'face_' + str(i+1)
    startX = faces[temp_str]['facial_area'][0]
    startY = faces[temp_str]['facial_area'][1]
    endX = faces[temp_str]['facial_area'][2]
    endY = faces[temp_str]['facial_area'][3]

    face = 

img1_faces = RetinaFace.extract_faces(img1_path)

for img1_face in img1_faces:
    plt.imshow(img1_face)
    plt.axis('off')
    plt.show()

    img1 = img1_face

### face detect and show by retinaface
img2_faces = RetinaFace.extract_faces(img2_path)

for img2_face in img2_faces:
    plt.imshow(img2_face)
    plt.axis('off')
    plt.show()

    obj = DeepFace.verify(img1, img2_face, model_name = model_name, enforce_detection = False)
    if (obj["distance"] <= 0.3) :
        print("same person")
        print(obj["distance"])
    else :
        print("different person")
    
    print(obj)
    print("------------------------")