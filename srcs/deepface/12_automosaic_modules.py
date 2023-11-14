from deepface import DeepFace
import cv2
from retinaface import RetinaFace

# giventime : /second
# return    : boolean, frame index
# usage     : cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
def detect_by_time(cap, given_time):
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = given_time * fps

    if (current_frame >= total_frames):
        return False, None
    return True, current_frame - 1

# save detected faces in given time
# save path should be end with /
# return : boolean (false if given time is out of video running time)
def capture_frame_faces(given_time, video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    res, frame = detect_by_time(cap, given_time)
    if (res == False):
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    res, frame = cap.read()
    retina_faces = RetinaFace.extract_faces(frame)

    idx = 0
    for retina_face in retina_faces:
        cv2.imwrite(save_path + "face_" + str(idx) + ".jpg", cv2.cvtColor(retina_face, cv2.COLOR_RGB2BGR)) # save image in RGB format, not BGR
        idx += 1

    return True

# mosaic video
# video_path : video path (should be end with /)
# image_path : image path (should be end with /)
# save_path  : save path (should be end with /)
# selected_image_num : number of images to compare
def mosaic_video(video_path, image_path, save_path, selected_image_num):
    cap = cv2.VideoCapture(video_path)
    v = 50 # mosaic size

    frame_width = int(cap.get(3))  # Width of the frames in the video
    frame_height = int(cap.get(4))  # Height of the frames in the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path + "output.mp4", fourcc, fps, (frame_width, frame_height)) # save video

    search_faces = list()

    for i in range(selected_image_num):
        load_image = image_path + "face_" + str(i) + ".jpg"
        img_faces = RetinaFace.extract_faces(load_image)
        for img_face in img_faces:
            search_faces.append(img_face)
    
    while(True):
        ret, frame = cap.read()
        if not ret: # if video is over
            break

        faces = RetinaFace.detect_faces(frame)
        for i in range(len(faces)):

            temp_str = 'face_' + str(i+1)
            startX = faces[temp_str]['facial_area'][0]
            startY = faces[temp_str]['facial_area'][1]
            endX = faces[temp_str]['facial_area'][2]
            endY = faces[temp_str]['facial_area'][3]

            unmosaic_flag = False

            for i in range(len(search_faces)):
                obj = DeepFace.verify(img1_path = search_faces[i], img2_path = frame[startY:endY, startX:endX], model_name = "VGG-Face", enforce_detection = False, detector_backend = 'retinaface')
                # modify detector_backend to retinaface
                if (obj["distance"] < 0.3) :
                    unmosaic_flag = True
                    break
            
            if (unmosaic_flag == False) :
                # mosaic
                roi_f = frame[startY:endY, startX:endX]
                roi = cv2.resize(roi_f, (roi_f.shape[1] // v, roi_f.shape[0] // v))
                roi = cv2.resize(roi, (roi_f.shape[1], roi_f.shape[0]), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = roi
        
        out.write(frame)

    cap.release()
    out.release()
    return True

def __main__():
    print("input video path : ")
    video_path = input()
    print("input image path : ")
    image_path = input()
    print("input save path : ")
    save_path = input()
    print("input selected image num : ")
    selected_image_num = int(input())

    print("input time to detect face (s) : ")
    given_time = int(input())

    if (capture_frame_faces(given_time, video_path, image_path) == False):
        print("Given time is out of video running time")
        exit()

    mosaic_video(video_path, image_path, save_path, selected_image_num)