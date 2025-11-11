"""
한국어 수어 단어 분류 모델 학습 스크립트 (개선 버전)

특징:
- Train/Validation 데이터 분할
- 학습 과정 로깅 및 시각화
- 체크포인트 자동 저장
- Early Stopping
- 학습률 스케줄러
- 성능 메트릭 (정확도, F1 스코어)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import sys
sys.path.append('../2_models/word_classifier')
from lstm_model import LSTMClassifier
import numpy as np
import os
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# ==================== 설정 ====================
class Config:
    # 데이터 경로
    data_dir = "../1_data/processed/sequence_data"
    checkpoint_dir = "../5_checkpoints"
    log_dir = "../logs"

    # 모델 하이퍼파라미터
    input_size = 84  # 양손 키포인트 (21*2*2)
    hidden_size = 256
    num_layers = 2
    dropout = 0.3
    bidirectional = True

    # 학습 하이퍼파라미터
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-5
    val_split = 0.2  # 검증 데이터 비율

    # Early Stopping
    patience = 10  # 개선이 없으면 N 에포크 후 종료

    # 기타
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)

# ==================== 데이터셋 ====================
class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.label_map = {}  # idx -> 단어명
        self.word_to_idx = {}  # 단어명 -> idx

        print(f"\n데이터 로딩 중: {data_dir}")

        # 단어별 폴더 탐색
        word_folders = sorted([f for f in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, f))])

        if not word_folders:
            raise ValueError(f"{data_dir}에 데이터가 없습니다!")

        for idx, word_name in enumerate(word_folders):
            self.label_map[idx] = word_name
            self.word_to_idx[word_name] = idx

            word_dir = os.path.join(data_dir, word_name)
            npy_files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]

            for npy_file in npy_files:
                keypoints = np.load(os.path.join(word_dir, npy_file))
                # 데이터 검증
                if keypoints.shape[1] != 84:
                    print(f"경고: {npy_file}의 특징 수가 {keypoints.shape[1]}입니다. 84를 기대했습니다.")
                    continue

                self.data.append(keypoints)
                self.labels.append(idx)

        print(f"총 {len(self.data)}개 샘플, {len(word_folders)}개 단어 로드됨")
        print(f"단어 목록: {word_folders}")

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # (seq_len, 84)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.label_map)

def collate_fn(batch):
    """가변 길이 시퀀스를 패딩하여 배치로 만듦"""
    xs, ys = zip(*batch)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # (batch, max_seq, 84)
    ys = torch.stack(ys)
    return xs_padded, ys

# ==================== 학습 함수 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="학습 중")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        # Forward
        pred = model(x)
        loss = criterion(pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 메트릭 계산
        total_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "acc": f"{100*correct/total:.2f}%"})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ==================== 메인 학습 루프 ====================
def main():
    # 시드 설정
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    print("=" * 60)
    print("한국어 수어 단어 분류 모델 학습 시작")
    print(f"디바이스: {Config.device}")
    print("=" * 60)

    # 데이터셋 로드
    full_dataset = SignDataset(Config.data_dir)
    num_classes = full_dataset.get_num_classes()

    # Train/Val 분할
    val_size = int(len(full_dataset) * Config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\n학습 데이터: {train_size}개")
    print(f"검증 데이터: {val_size}개")

    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # 모델 생성
    model = LSTMClassifier(
        input_size=Config.input_size,
        hidden_size=Config.hidden_size,
        num_layers=Config.num_layers,
        num_classes=num_classes,
        dropout=Config.dropout,
        bidirectional=Config.bidirectional
    ).to(Config.device)

    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate,
                            weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5, verbose=True)

    # 학습 기록
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # 학습 시작
    print(f"\n학습 시작! 총 {Config.num_epochs} 에포크")
    print("-" * 60)

    for epoch in range(Config.num_epochs):
        print(f"\n[Epoch {epoch+1}/{Config.num_epochs}]")

        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.device)
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, Config.device)

        # 학습률 스케줄러
        scheduler.step(val_loss)

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"학습 손실: {train_loss:.4f} | 학습 정확도: {train_acc:.2f}%")
        print(f"검증 손실: {val_loss:.4f} | 검증 정확도: {val_acc:.2f}%")

        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": full_dataset.label_map,
                "config": {
                    "input_size": Config.input_size,
                    "hidden_size": Config.hidden_size,
                    "num_layers": Config.num_layers,
                    "num_classes": num_classes,
                    "dropout": Config.dropout,
                    "bidirectional": Config.bidirectional
                }
            }

            save_path = os.path.join(Config.checkpoint_dir, "best_word_model.pth")
            torch.save(checkpoint, save_path)
            print(f"✓ 최고 모델 저장됨: {save_path}")
        else:
            patience_counter += 1
            print(f"검증 손실 개선 없음 ({patience_counter}/{Config.patience})")

        # Early Stopping
        if patience_counter >= Config.patience:
            print(f"\nEarly Stopping! {Config.patience} 에포크 동안 개선 없음")
            break

    # 학습 완료
    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"최고 검증 손실: {best_val_loss:.4f}")
    print("=" * 60)

    # 학습 곡선 저장
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.grid(True)

    plot_path = os.path.join(Config.log_dir, "training_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\n학습 곡선 저장됨: {plot_path}")

    # 학습 이력 JSON 저장
    history_path = os.path.join(Config.log_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"학습 이력 저장됨: {history_path}")

if __name__ == "__main__":
    main()
