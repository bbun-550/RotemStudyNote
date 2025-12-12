
# 📘 AI 모델 개발 과정 실습 저장소

`Data → ML → DL → CV → NLP → RAG` 까지 전 과정 실습 및 노트 정리

이 저장소는 AI 모델 개발 교육과정에서 수행한 모든 실습 코드, 학습 노트, 자료를 기술 스택별로 구조화하여 정리한 공간입니다.
각 스택은 폴더 내부 README.md와 연결되어 있어, GitHub에서 클릭하면 바로 해당 과목의 실습/노트를 볼 수 있습니다.

---

## 🧭 Repository Structure Overview

```yaml
.
├── 01_Data_Collection_Preprocessing
├── 02_Statistics
├── 03_AI_MLDL
├── 90.course_notes
├── 99.storage
└── README.md
```

---

# 🚀 Technical Stack Overview

## 1️⃣ Python Data Processing

**Skills**:
Python, Numpy, Pandas, Web Crawling, Data Preprocessing, Visualization(Matplotlib)

**Main Contents**:
- 데이터 수집
- 크롤링
- 전처리 파이프라인 구축
- 시각화 실습(Bar, Line, Scatter, Heatmap 등)

👉 [01_Data_Collection_Preprocessing](01_Data_Collection_Preprocessing)￼

---

## 2️⃣ Statistics & Classical Machine Learning

**Skills**:
Linear Regression, Logistic Regression,
Perceptron, Neural Network Basics,
Decision Tree, Random Forest, SVM,
Ensemble, Clustering(K-Means, Hierarchical),
PCA

**Main Contents**:
- 통계 기반 ML 알고리즘 구현
- 모델 평가 및 시각화
- 차원축소 실습

👉 [02_Statistics](02_Statistics)

---

### 3️⃣ Machine Learning & Deep Learning

**Skills**:
TensorFlow, Activation Functions,
Binary/Multiclass Classification,
Softmax, MNIST,
CNN, Dog vs Cat, CIFAR-10

**Main Contents**:
- 기본 신경망(NN)
- 합성곱 신경망(CNN) 모델 구현
- 이미지 분류 실습
- 모델 성능 비교 및 튜닝

👉 [03_AI_MLDL/01.tensorflow](03_AI_MLDL/01.tensorflow)

👉 [03_AI_MLDL/02.deeplearning](03_AI_MLDL/02.deeplearning)￼

---

## 4️⃣ Computer Vision (YOLO Series)

**Skills**:
YOLOv8, Object Detection,
Image Segmentation,
Model Export(ONNX/TensorRT)

**Main Contents**:
- 객체 탐지(Detection)
- 세그멘테이션
- 커스텀 데이터셋 학습
- 모델 변환(ONNX → TensorRT)
- 실전 이미지 처리 파이프라인 구축

👉 [03_AI_MLDL/03.yoloex](03_AI_MLDL/03.yoloex)

---

## 5️⃣ NLP (Natural Language Processing)

**Skills**:
NLP Basics,
Bag of Words, Tokenizer,
RNN, LSTM, Attention,
Self-Attention, Transformer

**Main Contents**:
- NLP 전처리
- RNN 기반 텍스트 분류
- LSTM 감정 분석
- Attention 메커니즘 구현 실습
- Transformer 구조 이해

👉 [03_AI_MLDL/04.nlp](03_AI_MLDL/04.nlp)￼

---

## 6️⃣ Vector Database / LangChain / RAG

**Skills**:
VectorDB (Chroma, FAISS),
LangChain, LangGraph,
Embeddings, RAG (Retrieval-Augmented Generation)

**Main Contents**:
- 벡터 임베딩 생성
- 문서 저장 & 검색 파이프라인 구축
- LangChain 기반 RAG 챗봇 구현
- 다양한 Retrieval 전략 실험

👉 [03_AI_MLDL/05.vectordb](03_AI_MLDL/05.vectordb)￼

👉 [03_AI_MLDL/06.lang](03_AI_MLDL/06.lang)

👉 [03_AI_MLDL/07.rag](03_AI_MLDL/07.rag)

---

## 📝 Course Notes

전체 수업 필기 및 정리 자료는 아래 폴더에 정리합니다.

👉 90.course_notes￼
- Daily Notes
- Code Explanation
- 참고문헌

---

## 🧪 Storage & Misc Codes

테스트 코드, 실습 이미지 및 데이터 소스 폴더입니다.

👉 99.storage￼
