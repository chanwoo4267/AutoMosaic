import cv2

face_cascade = cv2.CascadeClassifier('/Users/chanwoo4267/Downloads/deepface/opencv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/chanwoo4267/Downloads/deepface/opencv/haarcascade_eye.xml')

cap = cv2.VideoCapture('/Users/chanwoo4267/Downloads/deepface/sample_video/1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()

