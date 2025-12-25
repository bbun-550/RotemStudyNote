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
│── seq2seq.ipynb
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

👉 [NLP 기본 이론 정리](00.NOTES/251027_nlp.md)

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

👉 [NLP 기본 이론 정리](00.NOTES/251027_nlp.md)

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

👉 [RNN & LSTM 구조 이해와 시계열 학습 이론 정리](00.NOTES/251028_rnn.md)

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

### 📌 rnn03_sigmoid.ipynb — 시퀀스 기반 이진 분류와 Sigmoid 출력 구조

개요:\
**텍스트 데이터를 수치화**하여 **LSTM 기반 이진 분류(Binary Classification)를 수행**하는 전체 파이프라인를 다룬다.
**단어 사전 생성 → 토큰화 → 패딩 → 임베딩 → LSTM → Sigmoid** 출력까지,
자연어 분류 모델의 가장 기본적인 end-to-end 구조를 처음으로 완성하는 단계이다.

핵심 내용:
- **텍스트 수치화의 두 가지 방식 비교**
    - 문자열 처리 기반 수동 토큰화
        - 공백 기준 분리
        - 구두점 제거, 소문자 정규화
        - 단어 → 인덱스 사전 직접 생성
        
    - **Tokenizer 활용 자동 토큰화**
        - fit_on_texts()로 단어 사전 구축
        - texts_to_sequences()로 문장 → 정수 시퀀스 변환
        - texts_to_matrix()를 통한 BoW/TF-IDF 스타일 변환 확인
        
- **One-Hot Encoding 개념 재확인**
    - 토큰 시퀀스를 to_categorical()로 원-핫 벡터로 변환
    - “단어를 신경망 입력으로 쓰기 위해 반드시 수치화가 필요함”을 구조적으로 확인
    
- **시퀀스 길이 정규화 (Padding)**
    
    - 문장마다 길이가 다르기 때문에 RNN 입력 전 길이 통일 필요
    - `pad_sequences()` 사용
        - maxlen으로 최대 길이 지정
        - `padding='pre'` : 앞쪽 0 채우기
        
    - RNN/LSTM 입력은 **고정된 timesteps**를 요구한다는 점을 실습으로 체감
    
- **Embedding 레이어의 역할**
    
    - 단어 인덱스를 바로 LSTM에 넣지 않고, Embedding으로 변환
    - Embedding(input_dim, output_dim)
        - `input_dim`: 단어 사전 크기
            
        - `output_dim`: 단어 임베딩 차원
        
    - 단어를 **의미를 가진 밀집 벡터(Dense Vector)** 로 변환하는 계층
    
- **LSTM 기반 문장 분류 모델 구조**
    - 입력: 패딩된 정수 시퀀스
    - 흐름: 
`Input → Embedding → LSTM → Dense(ReLU) → Dense(Sigmoid)`
    - `LSTM(32, activation='tanh')`
        - 문장 전체 의미를 하나의 벡터로 요약 (many-to-one)
        
    - 중간 `Dense(ReLU)`
        - 비선형 표현력 강화
    
- **Sigmoid 출력층과 이진 분류**
    
    - 출력 노드 1개 + sigmoid 활성화
    - 출력값: 0~1 사이 확률
        - 긍정(1) / 부정(0) 분류
    - 손실 함수: `binary_crossentropy`
        - 이진 분류 문제에 특화
        
- **분류 결과 해석**
    - `model.predict()` → 확률 출력
    - 임계값 0.5 기준으로 클래스 결정
    

```python
np.where(pred > 0.5, 1, 0)
```

> **“신경망의 출력은 확률이며, 분류 기준은 사람이 정한다”는 점을 명확히 확인**

👉 [RNN 기반 텍스트 분류·생성 실습에서 새롭게 등장한 이론 정리](00.NOTES/251029_rnn.md)

---

### 📌 rnn04_softmax.ipynb — LSTM 기반 다항 분류와 Softmax를 활용한 단어 예측, 텍스트 생성


개요:
**LSTM을 이용한 다항 분류(Multi-class Classification)** 문제를 다루며,
하나의 입력 시퀀스로부터 **다음 단어를 예측**하고 이를 반복하여 **문장을 생성(Text Generation)** 하는 과정을 실습한다.

rnn03에서 다룬 이진 분류(Sigmoid)에서 확장되어,
**Softmax 출력층 + Categorical Crossentropy** 구조를 사용하는 것이 핵심이다.

---

핵심 내용:
- **문제 유형 전환: 이진 분류 → 다항 분류**
    - rnn03: 긍정/부정 (0 또는 1)
    - rnn04: 다음에 올 **단어 하나를 vocab 전체 중에서 선택**
    - 출력 클래스 수 = 단어 사전 크기(vocab_size)
- **Tokenizer 기반 단어 사전 구축**
    - 여러 문장을 하나의 코퍼스로 사용
    - `Tokenizer()` → 단어 단위 토큰화
    - word_index, index_word로 단어 ↔ 인덱스 매핑 확인   
    - `vocab_size = len(word_index) + 1`
- **다음 단어 예측용 학습 데이터 생성**
    - 문장을 점진적으로 늘려가며 시퀀스 생성
        - 예:      

```yaml
[경마장]
[경마장, 있는]
[경마장, 있는, 말이]
```

- -   
        
    - 입력(x): 이전 단어 시퀀스
    - 정답(y): 바로 다음 단
    - many-to-many 개념을 **데이터 구성 단계에서 구현**
    
- **Padding을 통한 시퀀스 길이 통일**    
    - 문장마다 길이가 다르므로 `pad_sequences()` 사용
    - padding='pre'로 앞쪽을 0으로 채움
    - 모델 입력을 하나의 고정된 텐서 형태로 맞춤
    
- **레이블 One-Hot Encoding**
    - 정답 단어(y)를 `to_categorical()`로 변환
    - 출력층 Softmax와 대응되는 형태
    - 각 샘플의 정답은 “단어 하나”지만, 벡터 차원은 vocab_size
    
- **모델 구조 (다항 분류 + 생성 기반)**

```yaml
Embedding → LSTM → Dense(ReLU) → Dense(Softmax)
```

- - `Embedding(vocab_size, 32, mask_zero=True)`
        - padding으로 들어간 0을 LSTM이 무시하도록 설정
        - 시퀀스 모델 안정성과 성능 향상
        
    - `LSTM(32, activation='tanh')`
        - 시퀀스 의미를 하나의 벡터로 요약
        
    - 출력층:        
        - `Dense(vocab_size, activation='softmax')`     
        - 각 단어가 다음에 나올 확률 분포 출력
        
    
- **손실 함수와 출력층의 대응**
    - Softmax 출력 → `categorical_crossentropy`
    - “확률 분포 간의 차이”를 최소화하도록 학습
    
- **텍스트 생성 로직 이해**
    - 현재 단어(또는 문장)를 입력으로 넣음
    - Softmax 결과 중 확률이 가장 높은 단어 선택 (argmax)
    - 선택된 단어를 다시 입력 문장 뒤에 붙임
    - 이 과정을 반복하여 문장 생성
    - **RNN/LSTM이 언어 모델(Language Model)처럼 동작함을 확인**
    
- **시드 단어(초기 단어)의 중요성**
    - 같은 모델이라도 시작 단어에 따라 생성 결과가 달라짐
    - “모델은 문법을 생성하는 것이 아니라, 학습된 확률 분포를 따라간다”는 점을 체감
<br>

**rnn03 ↔ rnn04 비교 요약:**

| **구분** | **sigmoid**         | **softmax**              |
| ------ | ------------------- | ------------------------ |
| 문제 유형  | 이진 분류               | 다항 분류                    |
| 출력 노드  | 1개                  | vocab_size               |
| 활성화 함수 | Sigmoid             | Softmax                  |
| 손실 함수  | Binary Crossentropy | Categorical Crossentropy |
| 목적     | 긍정/부정 판단            | 다음 단어 예측                 |
| 결과 활용  | 분류                  | 텍스트 생성                   |

👉 [RNN 기반 텍스트 분류·생성 실습에서 새롭게 등장한 이론 정리](00.NOTES/251029_rnn.md)

---

### 📌 rnn05_wordgen.ipynb — 뉴스 헤드라인 기반 LSTM 단어 생성(Language Model) 실습


개요:
실제 **신문 기사 헤드라인 데이터셋**을 활용하여
**LSTM 기반 언어 모델(Language Model)** 을 학습하고,
초기 단어(seed word)를 입력으로 받아 **자연스러운 문장을 생성**하는 과정을 실습한다.

rnn04가 개념 이해용 소규모 데이터였다면,
rnn05는 **실전 데이터 전처리 → 학습 데이터 구성 → 대규모 vocab → 문장 생성**까지
언어 모델의 전체 파이프라인을 경험하는 단계이다.

---

핵심 내용:

- **실제 데이터 로딩 및 탐색**
    - 뉴스 기사 헤드라인 CSV 로드
    - 총 샘플 수: 1,324 → 노이즈 제거 후 1,214
    - headline 컬럼만 추출하여 텍스트 데이터로 사용
    - Null은 아니지만 의미 없는 "Unknown" 값 제거
    
- **텍스트 전처리 (Preprocessing)**
    - 구두점 제거 (punctuation)
    - 소문자 통일
    - ASCII 외 문자 제거
    - 목적:
        - 단어 수(vocab) 폭발 방지            
        - 동일 단어의 표현 통합
    - 전처리 함수로 일괄 처리   
    
- **Tokenizer 기반 단어 집합(Vocabulary) 생성**    
    - Tokenizer().fit_on_texts(text)        
    - 전체 헤드라인 기반 단어 사전 구축        
    - vocab_size ≈ **3,494**        
    - 실제 언어 모델에서 vocab 크기가 얼마나 빠르게 커지는지 체감        
    
- **언어 모델 학습용 시퀀스 생성**    
    - 각 문장을 정수 시퀀스로 변환        
    - 문장 하나를 여러 개의 학습 샘플로 분해        
        - 예:    

```yaml
[i]
[i, disapprove]
[i, disapprove, of]
...
```

- -   
    - 입력(X): 이전 단어 시퀀스
    - 정답(y): 다음 단어
    - **다음 단어 예측 기반 언어 모델 구조**
- **Padding 및 데이터 분리**
    - 가장 긴 시퀀스 길이(max_len = 24) 기준으로 padding
    - `padding='pre'`
    - `X = sequences[:, :-1]`
    - `y = sequences[:, -1]`
    - y는 단어 하나 → 이후 **One-Hot Encoding** 필요
- **레이블 One-Hot Encoding**
    - `to_categorical(y, num_classes=vocab_size)`
    - 출력층 Softmax와 대응
    - 클래스 수 = vocab_size
- **LSTM 기반 언어 모델 구조**
    
```
Embedding → LSTM → Dense(ReLU) → Dense(Softmax)
```

- - `Embedding(vocab_size, 10)`
	- 단어 → 밀집 벡터 변환        
    - LSTM(128)
        - 문맥 정보 학습 (이전 단어들의 누적 의미)
    - 출력층:
        - vocab 전체에 대한 확률 분포 예측
    
- **손실 함수와 최적화**
    - Softmax + categorical_crossentropy
    - optimizer = adam
    - 언어 모델에서 가장 표준적인 조합
    
- **문장 생성 함수 구현**
    - 시작 단어(seed)를 입력
    - 현재 문장을 다시 모델 입력으로 사용
    - Softmax 결과 중 argmax 단어 선택
    - 예측 단어를 누적하여 문장 확장
    - 종료 조건:
        - 사전에 없는 단어
        - padding/미정 단어
        
    - 언어 모델의 **auto-regressive 생성 방식** 체험
    
- **결과 확인**
    - "i" → 비교적 문법적으로 자연스러운 문장 생성
    - "how" → 질문형 문장 생성
    - 데이터가 많아질수록 **문법·구조는 그럴듯하지만 의미는 불완전**한 특성 확인
        → 전통적 RNN 언어 모델의 한계 체감
    

📌 rnn04 ↔ rnn05 연결 포인트:

|**구분**|**rnn04**|**rnn05**|
|---|---|---|
|데이터|소규모 문장|실제 뉴스 헤드라인|
|vocab 크기|매우 작음|수천 단어|
|목적|개념 이해|실전 언어 모델|
|출력|다음 단어|문장 생성|
|체감 포인트|구조 이해|데이터 규모·성능 한계|

👉 [RNN 기반 텍스트 분류·생성 실습에서 새롭게 등장한 이론 정리](00.NOTES/251029_rnn.md)

---

### 📌 rnn06_char.ipynb — 문자(Character) 단위 LSTM 언어 모델 기초

개요:  
단어가 아닌 **문자(character) 단위 토큰화**를 사용하여 LSTM 기반 언어 모델을 학습하고,  
이전 문자 시퀀스를 바탕으로 **다음 문자를 예측**하는 과정을 실습한다.  
문자 단위 언어 모델의 데이터 구성 방식과 RNN/LSTM이 순차 패턴을 학습하는 방식을 이해하는 것이 목적이다.

핵심 내용:
- **Character-level Tokenization**
  - 텍스트 전체에서 고유 문자 집합(charset) 생성
  - 문자 ↔ 정수 인덱스 매핑 (`char_to_int`, `int_to_char`)
  - 단어 단위 대비 어휘 수(vocab)가 매우 작아짐
- **시퀀스 데이터 구성 방식**
  - 고정 길이(`seq_length`)의 문자 시퀀스를 입력으로 사용
  - 입력: 이전 N개의 문자
  - 출력: 바로 다음 문자 1개
  - 슬라이딩 윈도우 방식으로 학습 샘플 생성
- **One-Hot Encoding 기반 입력**
  - 문자 인덱스를 One-Hot 벡터로 변환
  - 입력 텐서 형태: `(samples, timesteps, vocab_size)`
- **LSTM 기반 문자 예측 모델**
  - 구조: `Input → LSTM → Dense(Softmax)`
  - 출력층 Softmax를 통해 다음 문자 확률 분포 예측
- **문자 단위 언어 모델의 특징**
  - 희귀 단어(OOV : Out of Vocabulary) 문제가 없음
  - 철자, 형태 패턴 학습에 강함
  - 의미 단위 이해는 제한적 (문법·의미는 약함)

👉 [Character-level Language Model & Text Generation 이론 정리](00.NOTES/251030_rnn.md)

---

### 📌 rnn07.ipynb — 문자 단위 LSTM 텍스트 생성과 샘플링 전략

개요:  
rnn06에서 학습한 **문자 단위 LSTM 언어 모델**을 기반으로,  
예측된 확률 분포를 활용해 **연속적인 텍스트를 생성**한다.  
단순 argmax 방식과 확률 샘플링 방식을 비교하며,  
텍스트 생성 품질에 영향을 주는 요소를 이해하는 것이 핵심이다.

핵심 내용:
- **Many-to-One 구조 기반 문자 예측**
  - 여러 문자 입력 → 다음 문자 1개 예측
  - 생성 과정에서는 예측 결과를 다시 입력으로 사용하는 auto-regressive 방식
- **텍스트 생성 루프 구현**
  - 초기 시드(seed) 문자 시퀀스 선택
  - 모델 예측 → 다음 문자 선택 → 입력 시퀀스 업데이트
  - 지정된 길이만큼 반복하여 문자열 생성
- **Sampling 전략의 필요성**
  - 단순 argmax:
    - 항상 가장 확률이 높은 문자 선택
    - 반복적이고 단조로운 결과 발생 가능
- **Temperature Sampling**
  - Softmax 확률 분포의 분산 조절
  - temperature ↓ : 보수적, 안정적, 반복적
  - temperature ↑ : 다양성 증가, 창의적이지만 노이즈 증가
- **Top-k Sampling**
  - 상위 k개 후보 문자만 남기고 나머지 확률 제거
  - 극단적으로 낮은 확률의 문자 선택 방지
- **결과 해석**
  - 문법적으로 완벽하지는 않지만,
    학습 데이터의 **스타일과 문자 패턴은 재현**
  - 문자 단위 모델의 한계와 가능성을 동시에 확인


👉 [Character-level Language Model & Text Generation 이론 정리](00.NOTES/251030_rnn.md)

---

### 📌 rnn08_toji.ipynb — 토지 소설 기반 문자 단위 LSTM 텍스트 생성

개요:\
대규모 한국어 텍스트(『토지』 소설)를 **문자(character) 단위**로 학습하여
LSTM 언어 모델을 구축하고, 확률 샘플링을 통해 **소설 스타일 텍스트를 생성**하는 실습이다.
**실전 데이터 + 대규모 설정**으로 확장한 단계에 해당한다.

핵심 내용:
- **대용량 텍스트 데이터 처리**    
    - 약 35만 자 분량의 소설 텍스트 로드
    - 한글 + 일부 문장부호(.,?!)만 유지하도록 정규식 정제
    - 불필요한 문자 제거를 통해 vocab 크기·메모리 관리
    
- **문자 집합(Charset) 구성**
    - 고유 문자 수: 약 1,400여 개 
    - char2idx, idx2char 매핑 직접 구성
    
- **슬라이딩 윈도우 기반 시퀀스 생성**
    - 입력 길이(maxlen = 30)
    - 샘플 간 간격(step = 10)을 두어 데이터 수/메모리 절충
    
- **정수 인덱스 기반 벡터화**
    - One-Hot 미사용
    - 입력: (N, maxlen) 정수 텐서
    - 출력: (N,) 다음 문자 인덱
    
- **Embedding + 다층 LSTM 모델**
    - Embedding으로 문자 인덱스를 밀집 벡터로 변환 → 메모리 절약   
    - LSTM 2층 + Dropout으로 표현력·일반화 강화
    
- **SparseCategoricalCrossentropy 사용**
    - 라벨을 One-Hot으로 만들지 않고 정수 인덱스 그대로 사용  
    - 대규모 vocab 환경에서 효율적인 학습
    
- **Temperature Sampling 기반 텍스트 생성**    
    - Softmax 확률 분포를 temperature로 조절
    - 완전히 무작위가 아닌 “그럴듯한 문장 흐름” 생성
    
- **결과 해석**
    - 문법·어휘는 소설 스타일을 모방
    - 의미 일관성은 제한적 → 문자 단위 모델의 한계 체감

---

### 📌 rnn09_jaso.ipynb — 자모 단위 LSTM 언어 모델과 한국어 생성

개요:\
한국어의 특성을 반영하여 **음절 → 자모(초성/중성/종성)** 단위로 분해한 뒤
LSTM 언어 모델을 학습하고, 다시 자모를 결합하여 **한글 문장을 생성**하는 실습이다.

문자 단위와 단어 단위 사이의 **한국어 특화 표현 방식**을 직접 비교/체험하는 단계이다.
 

핵심 내용:
- **자모 단위 토큰화**
    - `jamotools.split_syllables()`로 한글 음절을 자모로 분리
    - 생성 결과를 join_jamos()로 다시 음절 결합
    - 분리/결합 과정이 정보 손실 없이 동작함을 확인
    
- **자모 기반 Vocabulary 구성**
    - 자모 + 공백 + 기호 포함
    - vocab 크기: 약 180개 수준 → 문자 단위 대비 훨씬 작음
    - 희귀 토큰 처리를 위한 UNK 토큰 명시적 추가
    
- **tf.data.Dataset 기반 시퀀스 파이프라인**
    - 전체 자모 시퀀스를 고정 길이(`seq_length = 80`)로 분할
    - 입력: 이전 자모 시퀀스
    - 출력: 다음 자모 1개
    - shuffle + batch로 효율적 학습 구성
    
- **Embedding + LSTM + Softmax 구조**
    - Embedding으로 자모를 벡터화
    - 단일 LSTM으로 시퀀스 요약
    - Softmax로 다음 자모 확률 예측
    
- **Sparse Categorical Crossentropy 적용**
    - 자모 인덱스를 그대로 라벨로 사용
    
- **학습 중간 결과 출력**
    - Epoch 주기마다 생성 결과를 바로 출력
    - 학습이 진행될수록 “한국어스러운 형태”가 점점 나타남
    
- **자모 단위 생성의 특징**
    - 조사·어미 변화가 자연스럽게 나타나는 경우 증가
    - 철자·형태적 완성도는 문자 단위보다 우수
    - 여전히 의미·맥락은 불안정

---

### 📌 attention01.ipynb — Dot-Product Attention 메커니즘의 수치적 이해
 

개요:
\
Attention 메커니즘의 핵심 구조인 **Query–Key–Value(QKV)** 개념을 아주 작은 벡터 예제를 통해 **직접 계산**하는 실습.\
신경망 구현 이전에 Attention이 **“어디에 얼마나 집중하는가”를 어떻게 수치화하는지**를 내적(dot-product)과 softmax 연산으로 명확히 확인하는 것이 목적이다


핵심 내용:
- **Query–Key–Value(QKV) 구조의 역할 분리**
    - Query(Q): _현재 무엇에 집중할지 묻는 벡터_
    - Key(K): _입력 각 요소의 특징을 나타내는 벡터_
    - Value(V): _집중 결과로 실제 가져올 정보 벡터_
    
- **Dot-Product Attention의 핵심 연산 흐름**    
    1. Query와 Key의 내적 → 유사도 점수(scores)
    2. Softmax → Attention 가중치(weights)
    3. Value의 가중합 → Attention 출력(context vector)
    
- **Attention 가중치의 해석**
    - Softmax 결과는 “입력 요소별 중요도 분포”
    - 가중치의 합은 항상 1 → 확률적 해석 가능
    
- **Attention 출력 벡터의 의미**    
    - 단일 입력이 아니라, **여러 입력을 중요도에 따라 혼합한 요약 벡터**
    
- **수식이 아닌 숫자 계산을 통해**
    - “Attention은 선택이 아니라 가중 평균이다”라는 개념을 직관적으로 체감

👉 [Attention  이론 정리](00.NOTES/251104_attention.md)

---

### 📌 attention02.ipynb — Scaled Dot-Product Attention과 Context Vector의 역할

개요:
\
**Tokenizer + 시퀀스 + 위치별 Query**를 포함한
보다 실제 번역 구조에 가까운 **Scaled Dot-Product Attention 흐름** 실습.\
인코더–디코더 구조 없이도, Attention만으로 **문맥 기반 대응 관계**가 형성되는 과정을 확인한다.
  

핵심 내용:
- **Scaled Dot-Product Attention 도입**
    - 유사도 계산 시
        $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$    
    - 차원 수가 커질수록 점수가 커지는 문제를 완화하기 위한 스케일링        
    
- **Context Vector 개념의 명확화**    
    - Context Vector = Attention이 요약한 입력 정보        
    - 디코더는 각 출력 시점마다 **다른 Context Vector**를 가질 수 있음        
    
- **출력 위치별 Attention 분포 차이**    
    - 디코더의 각 타임스텝마다 Query가 달라짐        
    - 같은 입력이라도 출력 위치에 따라 집중 대상이 달라짐        
    
- **Alignment(정렬) 개념의 등장**    
    - Attention 가중치는 “출력 토큰 ↔ 입력 토큰” 간 대응 관계를 의미        
    - 번역 문제에서 단어 정렬(word alignment)의 직관적 표현        
    
- **인코더 없이도 Attention 원리 설명 가능**    
    - 실제 모델 대신 one-hot 벡터로 Key/Value 구성        
    - Attention의 **논리 구조 자체**에 집중한 설계        
    
- **Attention의 핵심 장점 확인**    
    - 모든 입력을 동시에 참고 가능        
    - RNN의 장기 의존성 문제를 구조적으로 해결할 수 있음을 시사        
    
👉 [Attention  이론 정리](00.NOTES/251104_attention.md)

---

### 📌 tformer01_selfAttention.ipynb — Transformer Self-Attention의 핵심 연산 구조 이해

개요:\
Transformer의 핵심 구성 요소인 Self-Attention 메커니즘을
아주 작은 숫자 벡터 예제를 통해 단계별로 직접 계산해보는 실습이다.
Query, Key, Value가 동일한 입력에서 만들어지는 Self-Attention의 연산 흐름과
각 토큰이 자기 자신을 포함해 다른 토큰을 어떻게 참조하는지를 직관적으로 이해하는 것이 목적이다.


핵심 내용:
- Self-Attention의 입력 구조
    - 3개의 토큰(A, B, C)을 2차원 임베딩 벡터로 표현
    - 입력 행렬 X shape = (num_tokens, embedding_dim)
    - 예:
        - A = [1, 0], B = [0, 1], C = [1, 1]
- Query / Key / Value의 생성
    - Self-Attention에서는 Q, K, V가 모두 동일한 입력 X에서 출발
    - 실제 Transformer에서는 Wq, Wk, Wv를 통해 선형 변환되지만, 본 실습에서는 개념 이해를 위해 단순화
- Scaled Dot-Product Attention 계산
    - Attention score 계산:
    $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
    - 각 토큰이 다른 모든 토큰과 얼마나 유사한지를 수치로 표현
    - 스케일링(√d_k)을 통해 값 폭주 방지
    - Softmax를 통한 Attention Weights
    - 각 Query 기준으로 score를 softmax
    - 결과:
        - 각 토큰이 모든 토큰에 얼마나 집중하는지를 나타내는 확률 분포
        - 행 단위 합 = 1
- Value의 가중합 → Self-Attention 출력
    - Attention weights × V
    - 결과 벡터는:
        - 자기 자신 + 다른 토큰 정보가 가중 평균된 새로운 표현
    - 출력 shape = (num_tokens, embedding_dim)
- Self-Attention의 의미
    - 각 토큰이 문맥 안에서 다른 토큰을 참조한 결과로 재표현
    - RNN 없이도 토큰 간 관계를 직접 모델링 가능
    - Transformer가 Seq2Seq를 대체할 수 있었던 핵심 이유를 수치적으로 확인

👉 [Transformer 이론 정리](00.NOTES/251105_transformer.md)

---

### 📌 tformer02_selfAttention.ipynb — 문장 수준 Self-Attention과 문맥(Context) 해석

개요:\
Self-Attention의 수식적 계산을 넘어서,
실제 문장 토큰 단위에서 Attention이 “어떤 단어를 참조하는지”를 해석하는 실습이다.
임의로 생성한 임베딩과 학습 가중치 행렬 $W_q$, $W_k$, $W_v$ 사용해
각 단어의 Attention 가중치 분포와 Context Vector를 직접 확인한다.
특히 대명사 “it”이 문맥 속에서 어떤 단어들에 의존하는지를 관찰함으로써,
Self-Attention이 참조 관계(coreference) 를 어떻게 포착하는지 직관적으로 이해 목적.

핵심 내용:
- 문장 단위 Self-Attention 실험 환경 구성
    - 입력 문장:
“The animal didn’t cross the street because it was too tired.”
    - 토큰 수: 12개
    - 각 단어를 4차원 임베딩 벡터(d_model = 4) 로 표현
    - 실제 학습 대신 랜덤 임베딩 + 랜덤 가중치 행렬로 Attention 구조만 시뮬레이션
    - Query / Key / Value의 “선형 변환” 의미 확인
    - 단순히 Q=K=V가 아니라:
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
- 같은 단어 임베딩이라도
    - Query: “내가 무엇을 보고 싶은가”
    - Key: “나는 어떤 정보를 가진 단어인가”
    - Value: “실제로 전달할 정보”
- 역할 분리의 필요성을 구조적으로 확인
- Scaled Dot-Product Attention의 실제 동작
    - 점수 계산:
$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$
- Softmax → Attention Weight Matrix
    - 각 행(row)은 하나의 Query(단어)가
모든 단어(Key)에 얼마나 집중하는지를 의미
    - 행 단위 합 = 1
    - 특정 단어(“it”)의 Attention 해석
    - “it” 토큰에 해당하는 attention 가중치 행을 추출
- 출력 예시:
    - street, tired, was 등에 상대적으로 높은 가중치
- 의미:
    - “it”이 문장 내에서
어떤 단어들을 문맥적으로 참조하고 있는지를 수치로 확인
- 대명사가 앞선 명사/상태를 참조하는 구조를
명시적 규칙 없이도 Attention으로 표현 가능
- Context Vector의 의미 명확화
    - Context Vector:
$$\text{context}_i = \sum_j \alpha_{ij} V_j$$
- “it”의 Context Vector는 주변 단어들의 Value 벡터를
Attention 가중치로 가중합한 결과
    - 즉, ‘it’이 문맥을 반영해 재해석된 표현
- RNN의 hidden state와 달리, 모든 입력 단어를 동시에 직접 참조
    - 단일 단어 관점에서의 Q / K / V 예시
    - “student”라는 임의 단어에 대해
    - Query / Key / Value 벡터를 각각 출력
- 목적:
    - “Q, K, V는 단순한 개념이 아니라
실제로는 서로 다른 선형 변환 결과”임을 수치로 확인
    - Transformer에서 표현 공간이 분리되는 이유를 직관적으로 이해
- tformer01과의 차별 포인트
- tformer01:
    - 작은 숫자 예제로 Self-Attention 수식 자체 이해
- tformer02:
    - 문장 단위 토큰
- Attention 가중치 해석
    - 대명사·문맥 참조 관점
    - Self-Attention이 “의미적 연결”을 만드는 방식을 강조

**정리 한 줄 요약:**

>Self-Attention은 단어 하나를 고립된 벡터로 처리하지 않고,
문장 내 모든 단어와의 관계를 가중치로 계산해
‘문맥을 반영한 새로운 표현(Context Vector)’으로 재구성한다.

👉 [Transformer 이론 정리](00.NOTES/251105_transformer.md)

---

### 📌 tformer03_selfAttention.ipynb — Multi-Head Attention 기반 Transformer Encoder와 문장 분류

개요:\
Self-Attention을 실제 모델 구조로 확장하여,
**Multi-Head Attention + Transformer Encoder** 블록을 이용해
문장 감정 분류(Sentiment Analysis)를 수행.

기존 RNN/LSTM 기반 분류와 달리,
순차 처리 없이도 문장 전체의 관계를 병렬적으로 학습할 수 있는
Transformer 인코더 구조의 실용적인 활용을 경험하는 것이 목적이다.
본 실습에서는 Seq2Seq 구조가 아닌, 인코더만 사용하는 Transformer가
분류 문제에 어떻게 적용되는지를 명확히 보여준다.

핵심 내용:
- Seq2Seq vs 문장 분류 문제 구조 구분
    - 번역·요약·텍스트 생성:
    → Encoder + Decoder 필요 (Seq2Seq)
    - 감정 분석·주제 분류·스팸 탐지:
    → 입력 문장 → 클래스 라벨 출력
    → Decoder 불필요, Encoder만 사용
    - Transformer는 문제 유형에 따라
Encoder-only / Encoder–Decoder로 유연하게 사용 가능함을 확인
- IMDB 데이터셋 기반 실전 분류 문제
    - 입력 데이터: 영화 리뷰 (정수 인덱스 시퀀스)
    - 출력 라벨: 긍정(1) / 부정(0)
    - 입력 전처리: 
        - 최대 단어 수 제한 (max_words = 10000)
        - 시퀀스 길이 통일 (pad_sequences, max_len = 100)
    - “텍스트 → 정수 시퀀스 → 모델 입력” 파이프라인 복습
- Transformer Encoder Block 구조 분해
    - 하나의 Encoder 블록 구성:
	    1.	Layer Normalization
	    2.	Multi-Head Self-Attention
	    3.	Residual Connection (잔차 연결)
	    4.	Feed Forward Network (Conv1D 기반)
	    5.	Residual Connection (2차)
    - 핵심 포인트:
        - Attention과 FFN 사이에 항상 잔차 연결
        - 깊은 네트워크에서도 gradient 흐름 유지
- Multi-Head Attention의 실질적 의미
    - num_heads = 4
    - 같은 문장을 서로 다른 4개의 관점(head) 에서 동시에 해석
    - 각 head는 서로 다른 관계 패턴(의존성, 강조 포인트)을 학습
    - 결과적으로:
        - 단일 Self-Attention보다 풍부한 표현 획득
- **Feed Forward Network**를 Conv1D로 구현한 이유
    - kernel_size = 1인 Conv1D는 Dense와 동일한 연산
    - 장점:
        - GPU 병렬 처리 효율 ↑
        - Dropout 삽입 용이
    - Transformer 논문의 FFN 구조를 실용적으로 변형한 구현 방식
- Position Encoding의 단순 대체 방식
    - 정식 Positional Encoding 대신:
        - Embedding 출력에 Gaussian Noise 추가
    - 목적:
        - 토큰 위치에 대한 미세한 차이 부여
        - 학습 안정성과 일반화 보조
    - “위치 정보가 완전히 없는 Self-Attention의 한계”를 완화하는 실험적 접근
- GlobalAveragePooling의 역할
    - Transformer Encoder 출력:
        - shape: (batch, timesteps, hidden_dim)
    - GlobalAveragePooling1D:
        - 모든 토큰 정보를 평균 내어
        - 문장 전체를 대표하는 하나의 벡터 생성
    - RNN의 “마지막 hidden state” 역할을 순서 의존 없이 대체
- **분류기(MLP Head) 구성**
    - Transformer Encoder 이후:
        - Dense(ReLU) + Dropout
        - Sigmoid 출력층
    - 출력 해석:
        - 0~1 사이 확률
        - 0.5 기준 → Positive / Negative
- Transformer 기반 분류 모델의 장점 체감
    - RNN/LSTM 대비:
    - 병렬 처리 가능 → 학습 속도 ↑
    - 장기 의존성 문제 구조적으로 해결
    - 문장 길이에 덜 민감
    - “Attention is All You Need”가 생성뿐 아니라 분류 문제에도 강력함을 확인
- 모델 성능 평가 시각화
    - 학습 곡선:
        - Accuracy / Loss (Train vs Validation)
    - Confusion Matrix:
        - Positive / Negative 분류 성능 직관적 확인
    - ROC Curve & AUC:
        - 이진 분류 모델의 전반적 판별력 평가
- Transformer가 실제 분류 문제에서도 안정적으로 동작함을 검증


**정리 한 줄 요약:**

Self-Attention을 “설명용 계산”에서 끝내지 않고,
**Multi-Head Attention + Transformer Encoder 구조**로 확장하여
RNN 없이도 문장 의미를 요약하고 분류할 수 있음을 증명.

👉 [Transformer 이론 정리](00.NOTES/251105_transformer.md)
