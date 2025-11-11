# 🤟 SLIVE v1.0.1 - 한국어 수어 통역기

**Sign Language Interpreter & Translator powered by AI**

실시간 웹캠을 통해 한국 수어를 인식하고 한글로 번역하는 딥러닝 기반 웹 애플리케이션입니다.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📺 데모

![수어 통역 데모](docs/demo.gif)

> 웹캠으로 수어를 인식하여 실시간으로 한글로 번역합니다.

---

## ✨ 주요 기능

- 🎥 **실시간 웹캠 수어 인식**: MediaPipe를 활용한 정확한 손 키포인트 추출
- 🧠 **딥러닝 모델**: 양방향 LSTM 기반 수어 단어 분류
- 🌐 **직관적인 웹 UI**: Streamlit 기반 사용자 친화적 인터페이스
- 📊 **학습 파이프라인**: Train/Val 분할, Early Stopping, 체크포인트 저장
- 📈 **시각화**: 학습 곡선, 정확도 그래프
- 🔄 **시퀀스 버퍼링**: 프레임 시퀀스 기반 정확한 인식
- 💾 **AI Hub 데이터셋 지원**: 한국 NIA 수어 데이터셋 호환

---

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.8 ~ 3.10
- 웹캠
- Windows / macOS / Linux

### 5분 안에 실행하기

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/SLIVE_v1.0.1.git
cd SLIVE_v1.0.1/SLIVE

# 2. 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 데이터 전처리 (샘플 데이터 포함)
python 1_data/utils/convert_json_to_sequence.py

# 5. 모델 학습
cd 4_training
python train_word_model_improved.py

# 6. 웹앱 실행
cd ../3_app
streamlit run streamlit_app.py
```

웹 브라우저가 자동으로 열립니다: `http://localhost:8501`

---

## 📁 프로젝트 구조

```
SLIVE_v1.0.1/
├── README.md                    # 프로젝트 소개
├── SLIVE/
│   ├── requirements.txt         # 패키지 목록
│   ├── QUICKSTART.md            # 빠른 시작 가이드
│   ├── WINDOWS_SETUP_GUIDE.md   # Windows 환경 설정 상세 가이드
│   │
│   ├── 1_data/                  # 데이터셋
│   │   ├── New_sample/          # 샘플 데이터
│   │   │   ├── 원천데이터/      # 원본 동영상 (.mp4)
│   │   │   └── LabelData/       # JSON 키포인트
│   │   ├── processed/           # 전처리된 데이터 (.npy)
│   │   └── utils/               # 데이터 처리 스크립트
│   │       └── convert_json_to_sequence.py
│   │
│   ├── 2_models/                # 모델 정의
│   │   ├── word_classifier/
│   │   │   └── lstm_model.py    # LSTM 분류 모델
│   │   └── seq2seq_translator/
│   │       └── seq2seq.py       # Seq2Seq 번역 모델
│   │
│   ├── 3_app/                   # 웹 애플리케이션
│   │   ├── streamlit_app.py     # Streamlit 웹앱 (신규!)
│   │   ├── realtime_infer.py    # OpenCV 기반 추론
│   │   └── lstm_model.py        # 모델 (로컬 복사본)
│   │
│   ├── 4_training/              # 학습 스크립트
│   │   ├── train_word_model_improved.py  # 개선된 학습 코드 (신규!)
│   │   ├── train_word_model.py  # 기본 학습 코드
│   │   └── lstm_model.py        # 모델 (로컬 복사본)
│   │
│   ├── 5_checkpoints/           # 학습된 모델
│   │   └── best_word_model.pth  # 최고 성능 모델
│   │
│   ├── 6_tests/                 # 테스트 코드
│   └── logs/                    # 학습 로그
│       ├── training_curve.png   # 학습 곡선
│       └── training_history.json
```

---

## 🎓 상세 가이드

### 전체 데이터셋으로 학습하기

1. **AI Hub 데이터셋 다운로드**
   - [AI Hub 한국어 수어 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=103) 접속
   - 회원가입 후 데이터 신청 (승인까지 1~2일 소요)
   - **원천데이터** + **라벨링데이터** 다운로드

2. **데이터 배치**
   ```
   SLIVE/1_data/New_sample/
   ├── 원천데이터/REAL/WORD/01/
   └── LabelData/REAL/WORD/01_real_word_keypoint/
   ```

3. **전처리 및 학습**
   ```bash
   # 데이터 전처리
   python 1_data/utils/convert_json_to_sequence.py

   # 학습 (전체 데이터: 30분~2시간)
   cd 4_training
   python train_word_model_improved.py
   ```

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| **딥러닝** | PyTorch, LSTM, Seq2Seq |
| **컴퓨터 비전** | MediaPipe, OpenCV |
| **웹 프레임워크** | Streamlit |
| **음성 합성** | gTTS (Google Text-to-Speech) |
| **데이터 처리** | NumPy, Pandas |
| **시각화** | Matplotlib |

---

## 🧠 모델 아키텍처

### LSTM 단어 분류 모델

```
입력: 손 키포인트 시퀀스 (batch, seq_len, 84)
  ↓
양방향 LSTM (256 hidden units, 2 layers)
  ↓
완전 연결층 (512 → 128 → num_classes)
  ↓
출력: 단어 클래스 확률
```

**특징:**
- 입력: 84 features (양손 21개 관절점 × 2 좌표 × 2)
- 양방향 LSTM으로 시간적 의존성 학습
- Dropout (0.3) 으로 과적합 방지
- 파라미터: ~424,000개

---

## 📊 성능

| 메트릭 | 값 |
|--------|-----|
| **정확도** (샘플 데이터) | ~85% |
| **추론 속도** | 30 FPS (CPU) |
| **모델 크기** | ~1.6 MB |

> ⚠️ 전체 데이터셋으로 학습 시 성능이 크게 향상됩니다.

---

## 🎯 사용 예시

### 웹앱 사용

1. **웹앱 실행**
   ```bash
   cd SLIVE/3_app
   streamlit run streamlit_app.py
   ```

2. **브라우저에서 `http://localhost:8501` 접속**

3. **사이드바 설정 조절**
   - 신뢰도 임계값: 0.7 (권장)
   - 시퀀스 길이: 30 프레임

4. **"🎥 시작" 버튼 클릭**

5. **수어 동작 시연**
   - 오른쪽에 인식 결과 실시간 표시
   - 히스토리에 번역 내역 누적

### OpenCV 기반 추론

```bash
cd SLIVE/3_app
python realtime_infer.py
```

---

## 🐛 문제 해결

### 자주 묻는 질문 (FAQ)

**Q: 웹캠이 열리지 않아요**
```
A: Windows 설정 → 개인정보 → 카메라 → 앱에서 카메라 액세스 허용
   다른 프로그램(Zoom, Teams)이 웹캠을 사용 중인지 확인
```

**Q: 패키지 설치 오류**
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 특정 버전으로 재설치
pip install mediapipe==0.10.0
pip install opencv-python==4.8.0.74
```

**Q: GPU 사용하고 싶어요**
```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

더 많은 문제 해결: [WINDOWS_SETUP_GUIDE.md](SLIVE/WINDOWS_SETUP_GUIDE.md)

---

## 📝 개발 로드맵

- [x] LSTM 기반 단어 분류 모델
- [x] Streamlit 웹 UI
- [x] 실시간 시퀀스 버퍼링
- [x] 학습 파이프라인 개선
- [ ] Seq2Seq 문장 번역
- [ ] Transformer 모델 적용
- [ ] 다국어 지원 (영어 수어)
- [ ] 모바일 앱 (Flutter)
- [ ] 클라우드 배포 (AWS/Azure)

---

## 🤝 기여하기

프로젝트에 기여하고 싶으신가요?

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👥 제작자

- **개발자**: Your Name
- **이메일**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)

---

## 🙏 감사의 말

- [AI Hub](https://www.aihub.or.kr/) - 한국어 수어 데이터셋 제공
- [MediaPipe](https://mediapipe.dev/) - 손 키포인트 추출
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Streamlit](https://streamlit.io/) - 웹 UI 프레임워크

---

## 📚 참고 자료

- [AI Hub 수어 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**⭐ 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

