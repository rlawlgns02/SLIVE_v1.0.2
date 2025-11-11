import cv2
import torch
import numpy as np
from lstm_model import LSTMClassifier
from gtts import gTTS
import os
import mediapipe as mp

model = LSTMClassifier()
model.load_state_dict(torch.load("5_checkpoints/word_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("웹캠을 켜고 수화를 해보세요. 'q' 누르면 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).flatten()
            x = torch.tensor(keypoints).float().view(1, 1, 63)
            pred = model(x)
            label = torch.argmax(pred, dim=1).item()
            word = f"단어{label}"
            cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            tts = gTTS(text=word, lang='ko')
            tts.save("temp.mp3")
            os.system("start temp.mp3")

    cv2.imshow('Sign Language Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
