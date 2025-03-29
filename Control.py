import cv2
import mediapipe as mp
from math import hypot
import pyvolume as vol
import numpy as np
from collections import deque
import math

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
previous_distance = None
distance_queue = deque(maxlen=5)  # Moving average window

# Calibration variables
min_distance = None
max_distance = None
calibrated = False

def get_distance(landmarkList):
    x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
    x_2, y_2 = landmarkList[8][1], landmarkList[8][2]
    return hypot(x_2 - x_1, y_2 - y_1)

def round_to_nearest(value, step):
    return round(value / step) * step

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList:
        distance = get_distance(landmarkList)
        if not calibrated:
            cv2.putText(frame, "Close your hand and press 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Open your hand and press 'o'", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if cv2.waitKey(10) & 0xff == ord('c'):
                min_distance = distance
                print(f"Min distance set to: {min_distance}")
            if cv2.waitKey(10) & 0xff == ord('o'):
                max_distance = distance
                print(f"Max distance set to: {max_distance}")
            if min_distance is not None and max_distance is not None:
                calibrated = True
                print("Calibration complete")
        else:
            distance_queue.append(distance)
            smoothed_distance = np.mean(distance_queue)
            
            if previous_distance is None or abs(smoothed_distance - previous_distance) >= 20:
                previous_distance = smoothed_distance
                # if frame_count % 1 == 0:
                b_level = math.ceil(np.interp(smoothed_distance, [min_distance, max_distance], [0, 100]))
                print(b_level)
                vol.custom(b_level)

    # frame_count += 1
    cv2.imshow('Image', frame)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()