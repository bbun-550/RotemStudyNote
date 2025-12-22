
# 📘 Attention  이론 정리

  

> attention01.ipynb, attention02.ipynb 실습을 통해
> **숫자 계산 수준에서 직접 확인한 Attention 메커니즘의 핵심 이론** 정리

---
## 1. Attention이 등장한 배경


RNN/LSTM 기반 시퀀스 모델에서는 다음과 같은 한계가 있었다.
- 인코더의 **마지막 hidden state 하나에 모든 정보를 압축**
- 긴 문장에서 **초반 정보 소실**
- 출력 시점마다 **다른 입력 위치에 집중하기 어려움**


이 문제를 해결하기 위해,
**출력 시점마다 입력 전체를 다시 참고하자**는 아이디어가 Attention이다.

👉 장기 의존성 문제의 배경 설명
\
👉[RNN/LSTM 한계](251028_rnn.md)

---

## 2. Query–Key–Value(QKV) 구조의 의미

Attention은 입력을 다음 세 역할로 분리해 사용

|**구성요소**|**의미**|**실습에서의 역할**|
|---|---|---|
|**Query (Q)**|지금 무엇을 찾고 싶은가|현재 출력 위치|
|**Key (K)**|각 입력이 어떤 특징을 갖는가|입력 토큰별 기준|
|**Value (V)**|실제로 꺼내 쓸 정보|입력 정보 자체|

> 핵심: **Q는 질문, K는 색인, V는 내용**

attention01에서는 **Q, K, V를 직접 숫자 벡터로 정의**해 계산을 추적

---

## 3. Dot-Product Attention - attention01.ipynb

### 3.1 유사도 계산 (Score)

Query와 Key의 유사도를 **내적(dot product)** 으로 계산

$$\text{score}_i = Q \cdot K_i$$

- 값이 클수록 해당 입력 토큰과 “더 관련 있음”
- 실습에서 scores = K.dot(Q) 로 직접 계산
    

---

### 3.2 Softmax → Attention Weights

유사도 점수는 확률 분포로 변환된다.  

$$\alpha_i = \text{softmax}(\text{score}_i)$$

- 모든 가중치의 합 = 1
- “어디에 얼마나 집중할지”를 수치로 표현
    

---

### 3.3 Value의 가중합 → Context Vector
  

$$\text{Context} = \sum_i \alpha_i V_i$$

- Attention의 **최종 출력**
- 입력 전체를 요약한 **문맥 벡터**
    

> 이 과정이 **Attention의 본질**

---

## 4. Scaled Dot-Product Attention - attention02.ipynb

  attention02에서는 **스케일링이 추가된 실제 Transformer 방식**을 사용

### 4.1 왜 $√dₖ$ 로 나누는가?

$$\text{score} = \frac{QK^T}{\sqrt{d_k}}$$

- 차원이 커질수록 dot product 값이 커짐    
- Softmax가 극단적으로 한 값에 쏠리는 현상 방지    
- **학습 안정성 향상**

> Transformer 수식의 핵심 **안정화 장치**

---

## 5. 출력 위치별 Attention 분포 (attention02의 핵심 관찰)

attention02에서는 **출력 위치마다 다른 Query**를 사용

- 출력 위치 0 → 입력 초반에 집중    
- 출력 위치 중간 → 중간 단어들에 집중    
- 출력 위치 마지막 → 입력 마지막에 집중    

> **Attention은 “출력 시점마다 다르게 본다”**
> 이 점이 RNN과의 결정적 차이.

---

## 6. Context Vector의 의미

Context Vector는 단순한 평균이 아니다.
- 입력 토큰들의 **의미 가중 평균**
- 출력 토큰을 생성할 때 참고하는 **동적 메모리**
  
> “문장을 하나로 요약한 벡터”가 아니라
> **“현재 출력에 최적화된 입력 요약 벡터”**

---

## 7. Attention과 Encoder–Decoder 연결 구조

attention02 실습은 실제 신경망을 생략했지만, 구조적으로는 다음을 모사한다.

```
[입력 문장]
   ↓
[Encoder Hidden States] → Key, Value
   ↓
[Decoder Hidden State]  → Query
   ↓
[Attention]
   ↓
[Context Vector]
   ↓
[출력 단어 생성]
```

👉 인코더/디코더 기본 구조
\
👉[Seq2Seq 개념](251222_seq2seq.md)

---

## 🔚 요약

- Attention은 **가중 평균 계산기**    
- Query가 바뀌면 **같은 입력도 다르게 해석**    
- 긴 문장에서도 정보 손실 없이 직접 참조 가능       
  
> Attention은
> **“출력 시점마다, 입력 전체를 다시 보고, 가장 중요한 정보만 뽑아 쓰는 메커니즘”** 이다.
