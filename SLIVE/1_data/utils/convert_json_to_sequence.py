import os
import json
import numpy as np
from tqdm import tqdm

# AI Hub 한국어 수어 데이터셋 전처리 스크립트
# JSON 키포인트 → NumPy 시퀀스 변환

SRC_DIR = "1_data/New_sample/LabelData/REAL/WORD/01_real_word_keypoint"
DST_DIR = "1_data/processed/sequence_data"

os.makedirs(DST_DIR, exist_ok=True)

print(f"SRC_DIR: {os.path.abspath(SRC_DIR)}")
print(f"DST_DIR: {os.path.abspath(DST_DIR)}")
print("\n데이터 전처리 시작...\n")

# 단어별로 그룹화 (WORD1501, WORD1502 등)
word_groups = {}
for label in os.listdir(SRC_DIR):
    label_dir = os.path.join(SRC_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    # NIA_SL_WORD1501_REAL01_D → WORD1501 추출
    word_id = label.split("_")[2]  # WORD1501
    if word_id not in word_groups:
        word_groups[word_id] = []
    word_groups[word_id].append((label, label_dir))

print(f"총 {len(word_groups)}개의 단어 발견")

# 각 단어별로 처리
for word_id, label_dirs in tqdm(word_groups.items(), desc="단어 처리"):
    word_save_dir = os.path.join(DST_DIR, word_id)
    os.makedirs(word_save_dir, exist_ok=True)

    # 각 방향(D, F, L, R, U)별로 처리
    for label, label_dir in label_dirs:
        keypoints_seq = []
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])

        for file in files:
            json_path = os.path.join(label_dir, file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # 양손 키포인트 추출 (왼손 + 오른손)
                    people = data.get("people", {})

                    # 왼손 키포인트 (x, y만 사용, confidence 제외)
                    left_hand = people.get("hand_left_keypoints_2d", [])
                    left_kps = []
                    if left_hand and len(left_hand) >= 63:
                        # x, y만 추출 (confidence 제외): 21개 × 2 = 42
                        for i in range(0, 63, 3):
                            left_kps.extend([left_hand[i], left_hand[i+1]])
                    else:
                        left_kps = [0.0] * 42

                    # 오른손 키포인트
                    right_hand = people.get("hand_right_keypoints_2d", [])
                    right_kps = []
                    if right_hand and len(right_hand) >= 63:
                        for i in range(0, 63, 3):
                            right_kps.extend([right_hand[i], right_hand[i+1]])
                    else:
                        right_kps = [0.0] * 42

                    # 전체 키포인트: 42 + 42 = 84 features
                    kps = left_kps + right_kps
                    keypoints_seq.append(kps)

            except Exception as e:
                print(f"    Error reading {json_path}: {e}")
                continue

        if keypoints_seq:
            keypoints_seq = np.array(keypoints_seq, dtype=np.float32)  # (프레임수, 84)
            save_path = os.path.join(word_save_dir, label + ".npy")
            np.save(save_path, keypoints_seq)
        else:
            print(f"    경고: {label_dir}에서 키포인트를 찾을 수 없습니다")

print(f"\n전처리 완료! 결과: {DST_DIR}")
print(f"총 {len(word_groups)}개 단어 처리됨")
