from deepface import DeepFace
import matplotlib.pyplot as plt

### set image path
img1_path = "/Users/chanwoo4267/Downloads/deepface/crowd_images/1.jpg"

### specify model
model_name = "VGG-Face"

### face detect and show example
img1_faces = DeepFace.extract_faces(img1_path)

for img1_face in img1_faces:
    print(img1_face)