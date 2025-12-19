## 📘 Natural Language Processing & Sequence Model 실습 정리

이 폴더는 자연어 처리(NLP)의 기초부터 순차 모델(RNN), Attention 메커니즘까지
언어 모델의 발전 흐름에 맞춰 단계적으로 학습하기 위한 실습 파일을 정리한 공간이다.
- NLP 기본 처리: 텍스트 전처리, 토큰화, 벡터화
- 단어 임베딩: Word2Vec을 통한 의미 공간 학습
- 순차 모델: RNN 기반 언어 모델과 단어/문자 생성
- Attention: RNN의 한계를 보완하는 핵심 개념


## 📂 Folder Structure

```yaml
04.nlp/
│── nlp01.ipynb
│── nlp02_word2vec.ipynb
│── nlp03_daum.ipynb
│── rnn01.ipynb
│── rnn02.ipynb
│── rnn03_sigmoid.ipynb
│── rnn04_softmax.ipynb
│── rnn05_wordgen.ipynb
│── rnn06_char.ipynb
│── rnn07.ipynb
│── rnn08_toji.ipynb
│── rnn09_jaso.ipynb
│── attention01.ipynb
│── attention01.ipynb
│── transformer01_selfAttention.ipynb
│── transformer02_selfAttention.ipynb
│── transformer03_selfAttention.ipynb
└── README.md
```


## 🔍 실습 파일별 개요

✏️ 필기 복습 중 ...

### 📌 nlp01.ipynb — NLP 기본 전처리와 토큰 단위 이해

개요:\
자연어 처리를 시작하기 위한 가장 기초 단계로, 텍스트 데이터를 컴퓨터가 처리 가능한 형태로 바꾸는 과정을 실습한다.

핵심 내용:
- 텍스트 데이터의 구조와 특징 이해
- 문장 → 단어 단위 분해(Tokenization)
- NLP 파이프라인에서 전처리의 중요성 인식


---

### 📌 nlp02_word2vec.ipynb — Word2Vec 기반 단어 임베딩과 유사도 분석

개요:\
단어를 단순한 인덱스가 아닌 의미를 반영한 벡터(Embedding) 로 표현하는 Word2Vec 모델을 실습한다.
단어 간 코사인 유사도, 유사 단어 탐색, 차원 축소 시각화를 통해
“단어의 의미가 벡터 공간에서 어떻게 표현되는지”를 직관적으로 이해한다.

핵심 내용:
- Word2Vec 개념 및 gensim 기반 모델 생성
- vector_size, window, sg(CBOW vs Skip-gram), alpha(learning rate) 이해
- 단어 → 밀집 벡터(Dense Vector) 변환 과정 확인
- 코사인 유사도를 통한 단어 간 의미적 거리 측정
- most_similar()를 이용한 유사 단어 탐색
- PCA를 활용한 고차원 단어 벡터의 2차원 시각화
- 벡터 각도(코사인 유사도 ↔ 각도)의 관계 해석
- 특정 단어 기준 유사도를 정렬해 텍스트 기반으로 비교 표현

---

### 📌 nlp03_daum.ipynb — 뉴스 기반 Word2Vec 실전 분석

개요:
뉴스를 활용하여 한국어 자연어 처리 파이프라인 전체 흐름을 실습한다.
형태소 분석을 통해 의미 있는 단어를 추출하고, Word2Vec으로 임베딩한 뒤 유사 단어 탐색, 벡터 연산, 시각화, 군집 분석까지 단계적으로 수행한다.

핵심 내용:
- KoNLPy(Okt)를 활용한 한국어 형태소 분석
- 명사·동사 중심 토큰 추출 및 불용 요소 제거
- 뉴스 텍스트 기반 단어 빈도 분석 및 CSV 저장
- Word2Vec 학습용 문장 파일(txt) 생성
- Skip-gram 기반 Word2Vec 모델 학습 및 저장/로드
- 한국어 단어 유사도 분석
- most_similar()를 활용한 의미적 유사 단어 탐색
- 벡터 덧셈/뺄셈을 통한 단어 의미 연산 실습
- PCA를 활용한 단어 임베딩 2차원 시각화
- 기준 단어 중심 의미 공간 확인
- KMeans 기반 단어 의미 군집화
- 군집 중심점 시각화 및 군집별 단어 해석
- 계층적 군집 분석(Dendrogram)을 통한 단어 관계 구조 파악

👉 [NLP 기본 이론 정리](00.NOTES/251027_nlp.md)

---

### 📌 rnn01.ipynb — 순환 신경망(RNN) 기본 구조와 입력·출력 형태 이해

개요:\
RNN 계열 모델(SimpleRNN, LSTM, GRU)의 네트워크 구조와 입력 텐서 형태를 이해하기 위한 기초 실습이다.
시계열 데이터가 모델에 어떻게 입력되고, 설정에 따라 출력 형태가 어떻게 달라지는지를 중심으로 학습한다.

핵심 내용:
- 순환 신경망(RNN)의 기본 개념과 시계열 처리 방식 이해
- RNN 계열 레이어(SimpleRNN, LSTM, GRU) import 및 비교
- 입력 텐서 형태 `(batch_size, timesteps, input_dim)` 구조 이해
- timesteps: 시퀀스(시간) 길이
- input_dim: 각 시점의 입력 벡터 크기
- SimpleRNN과 LSTM의 파라미터 수 계산 방식 차이
- SimpleRNN:
$(input_dim + units) × units + bias$
- LSTM:
$4 × (input_dim + units + 1) × units$ 
→ 4개의 게이트 구조로 인해 파라미터 수 증가
- batch_size를 명시적으로 지정했을 때 입력·출력 텐서 형태 확인
- 출력 형태에 따른 RNN 사용 방식 구분
- `return_sequences=False` (기본값)
→ many-to-one 구조 (마지막 시점 출력만 반환)
- `return_sequences=True`
→ many-to-many 구조 (모든 시점의 출력 반환)
- RNN 출력 형태 (batch_size, units) vs (batch_size, timesteps, units) 차이 이해

---

### 📌 rnn02.ipynb — LSTM을 활용한 시계열 숫자 예측 실습

개요:\
연속된 숫자 시퀀스를 입력으로 받아 다음 값을 예측하는 문제를 통해
LSTM(Long Short-Term Memory)의 동작 방식과 시계열 학습 과정을 실습한다.
단순 RNN 대비 LSTM이 왜 장기 의존성 문제에 강한지 직관적으로 이해하는 것이 목적이다.

핵심 내용:
- 시계열 데이터 구성
- 입력 x: 길이 3의 숫자 시퀀스
- 출력 y: 다음 시점의 값
- LSTM 입력을 위한 3차원 텐서 변환 `(samples, timesteps, features)`
  - 예: (8, 3, 1)
- LSTM 기반 회귀 모델 구성
  - `LSTM(units=10, activation='relu')`
  - 출력층: `Dense(1, activation='linear')`
- 손실 함수 및 최적화
  - `loss='mse'` (회귀 문제)
  - `optimizer='adam'`
- EarlyStopping을 활용한 과적합 방지
- 학습 데이터에 대한 예측값과 실제값 비교
- 새로운 시퀀스 입력에 대한 미래 값 예측
  - 예: [25, 35, 47] → 다음 값 예측


👉 [RNN & LSTM 구조 이해와 시계열 학습 이론 정리](00.NOTES/251028_rnn.md)

---

### 📌 rnn03_sigmoid.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn04_softmax.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn05_wordgen.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn06_char.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn07.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn08_toji.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rnn09_jaso.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 attention01.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 attention02.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 transformer01_selfAttention.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 transformer02_selfAttention.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 transformer03_selfAttention.ipynb — 

개요:

핵심 내용:
	•	
	•	

