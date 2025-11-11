# 🚀 빠른 시작 가이드

한국어 수어 통역 웹앱을 **5분 안에** 실행하는 방법입니다.

---

## ⚡ 빠른 설치 (Windows)

### 1. Python 설치 확인

```bash
python --version
# Python 3.8 ~ 3.10 권장
```

### 2. 가상 환경 생성 및 활성화

```bash
# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# CMD
python -m venv venv
venv\Scripts\activate.bat
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 📁 데이터 준비 (샘플 데이터로 테스트)

이미 프로젝트에 샘플 데이터가 포함되어 있습니다!

```bash
# 데이터 전처리
cd SLIVE
python 1_data/utils/convert_json_to_sequence.py
```

출력 예시:

```
데이터 전처리 시작...
총 3개의 단어 발견
전처리 완료!
```

---

## 🎓 모델 학습 (약 2~5분 소요)

```bash
cd 4_training
python train_word_model_improved.py
```

학습이 완료되면:

```
✓ 최고 모델 저장됨: ../5_checkpoints/best_word_model.pth
학습 완료!
```

---

## 🌐 웹앱 실행

```bash
cd ../3_app
streamlit run streamlit_app.py
```

브라우저가 자동으로 열립니다: `http://localhost:8501`

### 웹앱 사용법

1. **"🎥 시작"** 버튼 클릭
2. 웹캠 권한 허용
3. 손으로 수어 동작
4. 오른쪽에 인식 결과 표시!

---

## 🎯 전체 데이터셋으로 학습하기

### 1. AI Hub에서 데이터셋 다운로드

[AI Hub 한국어 수어 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)

- 회원가입 → 데이터 신청 → 승인 후 다운로드
- **원천데이터** + **라벨링데이터** 모두 필요

### 2. 데이터 배치

```
SLIVE/1_data/New_sample/
├── 원천데이터/REAL/WORD/01/
└── LabelData/REAL/WORD/01_real_word_keypoint/
```

### 3. 전처리 및 학습

```bash
# 전처리
python 1_data/utils/convert_json_to_sequence.py

# 학습 (전체 데이터는 30분~2시간 소요)
cd 4_training
python train_word_model_improved.py
```

---

## 🐛 문제 해결

### 웹캠이 안 열려요

```bash
# 다른 프로그램이 웹캠을 사용 중인지 확인 (Zoom, Teams 등)
# Windows 설정 → 개인정보 → 카메라 → 앱에서 카메라 액세스 허용
```

### 패키지 설치 오류

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 특정 버전으로 재설치
pip install mediapipe==0.10.0
pip install opencv-python==4.8.0.74
```

### PowerShell 실행 정책 오류

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📚 다음 단계

- **상세 가이드**: [WINDOWS_SETUP_GUIDE.md](./WINDOWS_SETUP_GUIDE.md)
- **프로젝트 구조**: [README.md](./README.md)
- **모델 개선**: 하이퍼파라미터 조정, Transformer 모델 시도

---

**즐거운 코딩 되세요! 🎉**
