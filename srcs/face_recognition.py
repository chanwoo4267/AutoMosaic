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

### face detect and show example
# img1 = DeepFace.detectFace(img1_path)
# img2 = DeepFace.detectFace(img2_path)
# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()

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


### face recognition
# resp1 = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name)
# resp2 = DeepFace.verify(img1_path = img1_path, img2_path = img3_path, model_name = model_name)
# resp3 = DeepFace.verify(img1_path = img1_path, img2_path = img4_path, model_name = model_name)
# resp4 = DeepFace.verify(img1_path = img1_path, img2_path = img5_path, model_name = model_name)
# resp5 = DeepFace.verify(img1_path = img1_path, img2_path = img6_path, model_name = model_name)
# resp6 = DeepFace.verify(img1_path = img1_path, img2_path = img7_path, model_name = model_name)
# resp7 = DeepFace.verify(img1_path = img1_path, img2_path = img8_path, model_name = model_name)
# resp8 = DeepFace.verify(img1_path = img1_path, img2_path = img9_path, model_name = model_name)
# resp9 = DeepFace.verify(img1_path = img1_path, img2_path = img10_path, model_name = model_name)

### print result
# print(resp1)
# print(resp2)
# print(resp3)
# print(resp4)
# print(resp5)
# print(resp6)
# print(resp7)
# print(resp8)
# print(resp9)