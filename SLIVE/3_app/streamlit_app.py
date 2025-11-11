"""
한국어 수어 통역 웹앱 (Streamlit)

실시간 웹캠을 통해 수어를 인식하고 한글로 번역합니다.
"""

import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
import sys
import os
import json

# 경로 추가
sys.path.append('../2_models/word_classifier')
from lstm_model import LSTMClassifier

# ==================== 설정 ====================
st.set_page_config(
    page_title="한국어 수어 통역기",
    page_icon="🤟",
    layout="wide"
)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==================== 모델 로드 ====================
@st.cache_resource
def load_model():
    """학습된 모델 로드"""
    checkpoint_path = "../5_checkpoints/best_word_model.pth"

    if not os.path.exists(checkpoint_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {checkpoint_path}")
        st.info("먼저 모델을 학습해주세요: python 4_training/train_word_model_improved.py")
        return None, None

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 모델 초기화
    config = checkpoint['config']
    model = LSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    )

    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 라벨 맵
    label_map = checkpoint['label_map']

    st.success(f"✓ 모델 로드 완료! (정확도: {checkpoint['val_acc']:.2f}%)")

    return model, label_map

# ==================== 키포인트 추출 ====================
def extract_keypoints(hand_landmarks):
    """손 랜드마크에서 키포인트 추출 (84 features)"""
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y])  # x, y만 사용 (z 제외)
    return keypoints  # 21 * 2 = 42

def extract_both_hands(results):
    """양손 키포인트 추출"""
    left_kps = [0.0] * 42
    right_kps = [0.0] * 42

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            kps = extract_keypoints(hand_landmarks)

            if label == "Left":
                left_kps = kps
            else:
                right_kps = kps

    return left_kps + right_kps  # 84 features

# ==================== 메인 앱 ====================
def main():
    st.title("🤟 한국어 수어 통역기")
    st.markdown("---")

    # 사이드바 설정
    st.sidebar.title("⚙️ 설정")
    confidence_threshold = st.sidebar.slider(
        "신뢰도 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="이 값보다 높은 신뢰도일 때만 결과를 표시합니다"
    )

    sequence_length = st.sidebar.slider(
        "시퀀스 길이 (프레임)",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="수어 인식에 사용할 프레임 수"
    )

    show_keypoints = st.sidebar.checkbox("키포인트 표시", value=True)

    # 모델 로드
    model, label_map = load_model()
    if model is None:
        st.stop()

    # 컬럼 레이아웃
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 웹캠 피드")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📝 인식 결과")
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()

        st.subheader("📜 히스토리")
        history_placeholder = st.empty()

    # 시작/중지 버튼
    start_button = st.button("🎥 시작", type="primary")
    stop_button = st.button("⏹️ 중지")

    if start_button:
        st.session_state['running'] = True

    if stop_button:
        st.session_state['running'] = False

    # 세션 상태 초기화
    if 'running' not in st.session_state:
        st.session_state['running'] = False

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # 웹캠 실행
    if st.session_state['running']:
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 시퀀스 버퍼
        sequence_buffer = deque(maxlen=sequence_length)

        try:
            while st.session_state['running']:
                ret, frame = cap.read()
                if not ret:
                    st.error("웹캠을 열 수 없습니다")
                    break

                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 손 인식
                results = hands.process(frame_rgb)

                # 키포인트 추출
                if results.multi_hand_landmarks:
                    # 양손 키포인트
                    keypoints = extract_both_hands(results)
                    sequence_buffer.append(keypoints)

                    # 손 그리기
                    if show_keypoints:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                frame_rgb,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )

                    # 시퀀스가 충분히 쌓이면 추론
                    if len(sequence_buffer) == sequence_length:
                        # 시퀀스 준비
                        seq = torch.tensor(list(sequence_buffer), dtype=torch.float32)
                        seq = seq.unsqueeze(0)  # (1, seq_len, 84)

                        # 추론
                        with torch.no_grad():
                            pred = model(seq)
                            probs = torch.softmax(pred, dim=1)
                            confidence, predicted_idx = torch.max(probs, 1)

                        confidence_val = confidence.item()
                        predicted_idx_val = predicted_idx.item()

                        # 결과 표시 (임계값 이상일 때만)
                        if confidence_val >= confidence_threshold:
                            word = label_map[str(predicted_idx_val)]

                            # 화면에 결과 표시
                            cv2.putText(
                                frame_rgb,
                                f"{word} ({confidence_val:.2%})",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 255, 0),
                                3
                            )

                            # 사이드바 결과 업데이트
                            result_placeholder.markdown(f"## {word}")
                            confidence_placeholder.progress(confidence_val)

                            # 히스토리 추가 (중복 방지)
                            if not st.session_state['history'] or st.session_state['history'][-1] != word:
                                st.session_state['history'].append(word)
                                if len(st.session_state['history']) > 10:
                                    st.session_state['history'].pop(0)

                            history_placeholder.write(" → ".join(st.session_state['history']))
                        else:
                            cv2.putText(
                                frame_rgb,
                                "낮은 신뢰도",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2
                            )

                # 프레임 표시
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        finally:
            cap.release()
            hands.close()

    else:
        st.info("👆 '시작' 버튼을 클릭하여 수어 통역을 시작하세요")

# ==================== 앱 실행 ====================
if __name__ == "__main__":
    main()
