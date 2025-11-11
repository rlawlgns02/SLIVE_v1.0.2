import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    한국어 수어 단어 분류 모델

    Args:
        input_size: 입력 특징 수 (양손 키포인트: 84 = 21*2*2)
        hidden_size: LSTM 은닉층 크기
        num_layers: LSTM 레이어 수
        num_classes: 분류할 단어 개수
        dropout: 드롭아웃 비율 (과적합 방지)
        bidirectional: 양방향 LSTM 사용 여부
    """
    def __init__(self, input_size=84, hidden_size=256, num_layers=2,
                 num_classes=100, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 양방향인 경우 출력 크기가 2배
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # 완전 연결층
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            out: (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, (hn, cn) = self.lstm(x)

        # 양방향인 경우 마지막 hidden state 연결
        if self.bidirectional:
            # hn: (num_layers*2, batch, hidden_size)
            hn_forward = hn[-2]  # 마지막 레이어의 forward 방향
            hn_backward = hn[-1]  # 마지막 레이어의 backward 방향
            hn_last = torch.cat([hn_forward, hn_backward], dim=1)
        else:
            hn_last = hn[-1]

        # 완전 연결층 통과
        out = self.fc(hn_last)
        return out
