import cv2

face_cascade = cv2.CascadeClassifier('/Users/chanwoo4267/Downloads/deepface/opencv/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4')
v = 50

frame_width = int(cap.get(3))  # Width of the frames in the video
frame_height = int(cap.get(4))  # Height of the frames in the video
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
   
        roi = cv2.resize(roi_color, (w // v, h // v))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        frame[y:y+h, x:x+w] = roi

    out.write(frame)
    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
out.release()
cv2.destroyAllWindows()