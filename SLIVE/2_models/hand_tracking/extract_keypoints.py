import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=6)  # max_num_hands=2로 변경
mp_draw = mp.solutions.drawing_utils

SAVE_DIR = '1_data/processed/keypoints'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print("웹캠이 켜졌습니다. 손을 화면에 보여주세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):  # enumerate로 인덱스 추가
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).flatten()

            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_hand{idx}.npy"  # 손 인덱스 추가
            np.save(os.path.join(SAVE_DIR, filename), keypoints)

    cv2.imshow('Hand Keypoints', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
