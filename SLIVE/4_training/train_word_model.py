import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lstm_model import LSTMClassifier
import numpy as np
import os

class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.label_map = {}
        for idx, label in enumerate(sorted(os.listdir(data_dir))):
            self.label_map[idx] = label
            for file in os.listdir(os.path.join(data_dir, label)):
                keypoints = np.load(os.path.join(data_dir, label, file))
                self.data.append(keypoints)
                self.labels.append(idx)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).float()  # shape: (프레임수, 63)
        y = torch.tensor(self.labels[idx]).long()
        return x, y

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # (batch, max_seq, 63)
    ys = torch.stack(ys)
    return xs_padded, ys

dataset = SignDataset("1_data/processed/sequence_data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = LSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for x, y in dataloader:
        # x: (batch, seq_len, 63)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{epoch+1} epoch loss: {loss.item():.4f}")

torch.save(model.state_dict(), "5_checkpoints/word_model.pth")
