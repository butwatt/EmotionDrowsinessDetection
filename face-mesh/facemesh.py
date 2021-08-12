import mediapipe as mp
import numpy as np
import cv2
from scipy.spatial import distance as dist

leftEye_l = 263
leftEye_r = 362
leftEye_t = 386
leftEye_b = 374

rightEye_l = 133
rightEye_r = 33
rightEye_t = 159
rightEye_b = 145

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

cap = cv2.VideoCapture(0)

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw=mp.solutions.drawing_utils



while True:
    _, frm = cap.read()
    # print(frm.shape)
    blank_frm = np.zeros((480, 640, 3), np.uint8)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = face.process(rgb)
    # print(dir(op))
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            # print(i.landmark)
            ear=eye_aspect_ratio(i.landmark)
            # print(ear)
            # draw.draw_landmarks(frm, i, facmesh.FACE_CONNECTIONS, landmark_drawing_spec=draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255,255)))
            cv2.putText(blank_frm, "EAR: {:.2f}".format(ear), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            draw.draw_landmarks(blank_frm, i, facmesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255)))

    cv2.imshow("window",frm)
    cv2.imshow("facemesh", blank_frm)

    if cv2.waitKey(1) ==27:
        cap.release()
        cv2.destroyAllWindows()
        break