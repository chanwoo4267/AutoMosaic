from deepface import DeepFace
import cv2
from retinaface import RetinaFace

# video_path = "/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4"
# img1_path = "/Users/chanwoo4267/Downloads/deepface/sample_video/2.png"
# img2_path = "/Users/chanwoo4267/Downloads/deepface/sample_video/3.png"

video_path = "../../sample_video/1.mp4"
img1_path = "../../sample_video/2.png"
img2_path = "../../sample_video/3.png"

cap = cv2.VideoCapture(video_path)
v = 50

frame_width = int(cap.get(3))  # Width of the frames in the video
frame_height = int(cap.get(4))  # Height of the frames in the video
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

img1_faces = RetinaFace.extract_faces(img1_path)
for img1_face in img1_faces:
    img1 = img1_face

img2_faces = RetinaFace.extract_faces(img2_path)
for img2_face in img2_faces:
    img2 = img2_face

unmosaic = [img1, img2]

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

        unmosaic_flag = False

        for i in range(len(unmosaic)):
            obj = DeepFace.verify(img1_path = unmosaic[i], img2_path = frame[startY:endY, startX:endX], model_name = "VGG-Face", enforce_detection = False, detector_backend = 'retinaface')
            # modify detector_backend to retinaface
            if (obj["distance"] < 0.3) : # 같은사람
                unmosaic_flag = True
                break # 더이상 체크할 필요 없음
            
        if (unmosaic_flag == False) :
            # 모자이크 처리
            roi_f = frame[startY:endY, startX:endX]
            roi = cv2.resize(roi_f, (roi_f.shape[1] // v, roi_f.shape[0] // v))
            roi = cv2.resize(roi, (roi_f.shape[1], roi_f.shape[0]), interpolation=cv2.INTER_AREA)
            frame[startY:endY, startX:endX] = roi
    
    out.write(frame)
    # cv2.imshow('result', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
out.release()
cv2.destroyAllWindows()
