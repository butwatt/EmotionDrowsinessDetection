# This file is going to receive realtime webcam images and detect faces, emotion,
# and sleepiness using open source code and facial landmarking

import mediapipe as mp
import numpy as np
import cv2
from scipy.spatial import distance as dist
from threading import Thread
import playsound
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

leftEye_l = 263
leftEye_r = 362
leftEye_t = 386
leftEye_b = 374

rightEye_l = 133
rightEye_r = 33
rightEye_t = 159
rightEye_b = 145

EYE_AR_THRESH = 0.3
SLEEPY_CONSEC_FRAMES = 5
PRINT_SLEEPY_FRAMES = 20
# initialize the frame counters and the total number of blinks
EYE_SHUT_COUNTER = 0
TOTAL = 0
Alpha = 0.01
ear_fma = 0.3
frameN = 0
# learning time 1000 frame
CALIBRATION_FRAMES = 100
print_sleep = False
print_sleep_count = 0
ALARM_ON = False
SLEEPY_THRESHOLD = 0.8

cv2.namedWindow("Face Display", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Facemesh Display", cv2.WINDOW_KEEPRATIO)

def sound_alarm():
    path = "sound/DogWoof.mp3"
    playsound.playsound(path)
    global ALARM_ON
    ALARM_ON = False

def eye_aspect_ratio(landmark):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    leftEye_l_XY = [landmark[leftEye_l].x, landmark[leftEye_l].y]
    leftEye_r_XY = [landmark[leftEye_r].x, landmark[leftEye_r].y]
    leftEye_t_XY = [landmark[leftEye_t].x, landmark[leftEye_t].y]
    leftEye_b_XY = [landmark[leftEye_b].x, landmark[leftEye_b].y]
    Horizontal_left = dist.euclidean(leftEye_l_XY, leftEye_r_XY)
    vertical_left = dist.euclidean(leftEye_t_XY, leftEye_b_XY)
    left_ratio = vertical_left/Horizontal_left

    rightEye_l_XY = [landmark[rightEye_l].x, landmark[rightEye_l].y]
    rightEye_r_XY = [landmark[rightEye_r].x, landmark[rightEye_r].y]
    rightEye_t_XY = [landmark[rightEye_t].x, landmark[rightEye_t].y]
    rightEye_b_XY = [landmark[rightEye_b].x, landmark[rightEye_b].y]
    Horizontal_right = dist.euclidean(rightEye_l_XY, rightEye_r_XY)
    vertical_right = dist.euclidean(rightEye_t_XY, rightEye_b_XY)
    right_ratio = vertical_right / Horizontal_right

    # compute the eye aspect ratio
    ear = (left_ratio + right_ratio)/2
    # return the eye aspect ratio
    return ear

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_detector = load_model('EmotionDetectionModel_Andrew911-0.67.h5')
print(emotion_detector.summary())
class_labels=['Angry','Happy','Neutral','Sad','Surprise','Sleepy']

#little note: these colors are in BGR
neutralcolor=(255, 255 ,255)
angrycolor=(0, 0, 233)
happycolor=(0, 246, 0)
surprisecolor=(0, 255, 251)
sadcolor=(255, 104, 0)
sleepycolor=(143, 39, 146)
nullcolor=(100, 100, 100)

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw=mp.solutions.drawing_utils

while True:

    frameN = frameN + 1
    #print("frameN", frameN)

    labels = []

    _, frm = cap.read()
    # print(frm.shape)
    blank_frm = np.zeros((480, 640, 3), np.uint8)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    emotion_number = -1
    for (x,y,w,h) in faces:
        #cv2.rectangle(frm,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=emotion_detector.predict(roi)[0]
            #print(preds.argmax())
            emotion_number = preds.argmax()

            if print_sleep == True:
                emotion_number = 5

            label=class_labels[emotion_number]
            #print(label)
            label_position=(x,y)
            cv2.putText(frm,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
        else:
            cv2.putText(frm,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)\

    op = face.process(rgb)
    # print(dir(op))
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            #print(i.landmark)
            # print(ear)


            #mod = frameN % 255
            if emotion_number == 0:
                mycolor = angrycolor
                mythickness = 5
            elif emotion_number == 1:
                mycolor = happycolor
                mythickness = 3
            elif emotion_number == 2:
                mycolor = neutralcolor
                mythickness = 1
            elif emotion_number == 3:
                mycolor = sadcolor
                mythickness = 4
            elif emotion_number == 4:
                mycolor = surprisecolor
                mythickness = 5
            elif emotion_number == 5:
                mycolor = sleepycolor
            else:
                mycolor = nullcolor
                mythickness = 1

            draw.draw_landmarks(blank_frm, i, facemesh.FACE_CONNECTIONS, landmark_drawing_spec=draw.DrawingSpec(thickness=mythickness, circle_radius=1, color=mycolor))
            draw.draw_landmarks(frm, i, facemesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255)))

        ear=eye_aspect_ratio(i.landmark)
        if frameN < CALIBRATION_FRAMES:
            # update ear_fma
            if frameN < 2:
                ear_fma = ear
            else:
                ear_fma = ear_fma * (1 - Alpha) + ear * Alpha;
            cv2.putText(frm, "Calibrating...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        else:
            if ear < ear_fma * SLEEPY_THRESHOLD:
                EYE_SHUT_COUNTER += 1
                if EYE_SHUT_COUNTER >= SLEEPY_CONSEC_FRAMES:
                    print_sleep = True
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()
            else:
                # reset the eye frame counter
                EYE_SHUT_COUNTER = 0


        if print_sleep:
            print_sleep_count = print_sleep_count + 1
            if print_sleep_count < PRINT_SLEEPY_FRAMES:
                cv2.putText(frm, "Stop sleeping!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if print_sleep_count == PRINT_SLEEPY_FRAMES:
                print_sleep = False
                print_sleep_count = 0

        cv2.putText(frm, "EAR_FMA: {:.2f}".format(ear_fma), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frm, "EAR: {:.2f}".format(ear), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Display",frm)
    cv2.imshow("Facemesh Display", blank_frm)
    if cv2.waitKey(1) == 32:
        frameN = 0

    if cv2.waitKey(1) ==27:
        cap.release()
        cv2.destroyAllWindows()
        break