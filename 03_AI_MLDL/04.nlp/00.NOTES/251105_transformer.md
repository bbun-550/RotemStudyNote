# 📘 Transformer 이론 정리

## 1. Transformer 등장 배경과 핵심 전환점

**Transformer**는 순차 처리에 의존하던 RNN/LSTM 계열 구조를 완전히 벗어나,
**Attention만으로 시퀀스를 처리**하는 구조를 제안했다.

- RNN/LSTM    
    - 시점별 순차 계산 → 병렬화 어려움        
    - 장기 의존성 문제 구조적으로 존재        
        👉 [RNN & LSTM 장기 의존성](251028_rnn.md)
    
- Transformer    
    - 모든 토큰을 **동시에 처리**        
    - 시점 간 거리와 무관하게 직접 참조 가능        
    - 병렬 연산 최적화 → 학습 속도 대폭 개선        

> **핵심 전환:**
> “시간축(time-step)” 중심 → “관계(relationship)” 중심 모델

---

## 2. Self-Attention의 구조적 의미

Self-Attention은 **입력 시퀀스 내부의 토큰들 간 관계를 직접 계산**한다.

### 2.1 기존 Attention과의 차이

- 기존 Attention:    
    - Encoder–Decoder 간 상호작용 중심        

- Self-Attention:    
    - **입력 내부에서 자기 자신을 포함한 모든 토큰을 참조**        
  
👉 Attention 기본 개념
\
👉 [Attention 기본 구조](251104_attention.md)

### 2.2 Self-Attention의 본질

- 각 토큰은:    
    - **Query:** “내가 무엇을 보고 싶은가”        
    - **Key:** “나는 어떤 정보를 가진 토큰인가”        
    - **Value:** “실제로 전달할 정보”        
    
- 결과:    
    - 모든 토큰이 **문맥을 반영한 새로운 표현**으로 재구성됨

> Self-Attention은
> **“토큰을 고정된 의미 벡터가 아니라, 문맥에 따라 변하는 표현”으로 만든다.**

---

## 3. Q, K, V 분리의 이유

tformer02 실습에서는 **Q, K, V가 서로 다른 선형 변환 결과**임을 수치적으로 확인했다.

### **3.1 왜 굳이 나누는가?**

같은 임베딩 X에서 출발하더라도,
- Query: “내 관점에서”    
- Key: “상대 토큰의 특성”    
- Value: “실제 정보”    

>**서로 다른 공간**으로 투영해야 **관계 해석의 자유도가 커진다.**

### 3.2 효과

- 같은 단어라도:    
    - 질문하는 입장(Query)        
    - 참조되는 입장(Key)        
    - 정보 제공자(Value)        
        가 분리됨       
    
- 문법·의미·지시 관계(coreference)를 동시에 포착 가능

> 이 분리는 Transformer가
> **단순 유사도 계산을 넘어 ‘역할 기반 관계 학습’을 가능하게 하는 핵심 설계**다.

---

## 4. Scaled Dot-Product Attention의 안정성 의미

Self-Attention에서 사용하는 점수는 다음과 같다:

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$
### **왜 스케일링이 필요한가?**

- 차원 d_k 가 커질수록 내적 값 폭증    
- Softmax 입력이 커지면:    
    - 특정 토큰에만 극단적으로 집중        
    - gradient 소실 위험 증가        

### $√dₖ$의 역할

- score 범위를 안정화    
- 학습 초반부터 균형 잡힌 attention 분포 유지    


> 스케일링은 성능 향상이 아니라
> **학습 안정성을 위한 필수 설계 요소**

---

## 5. Multi-Head Attention의 정보 확장 효과

### 5.1 단일 Head의 한계

- 하나의 관계 패턴만 포착
- 특정 관점에 과도하게 치우칠 수 있음

### 5.2 Multi-Head 구조

- 여러 개의 Attention Head를 병렬로 수행    
- 각 Head는:    
    - 서로 다른 관계 패턴 학습        
    - 서로 다른 표현 공간 탐색        
    
예:
- Head 1: 주어–동사 관계    
- Head 2: 감정 표현    
- Head 3: 부정·강조 표현    
- Head 4: 문장 후반부 요약    

> Multi-Head Attention = **“하나의 문장을 여러 관점에서 동시에 해석”**

---

## 6. Transformer Encoder Block의 구조적 완성도

tformer03에서 구현한 Encoder Block은 논문 구조를 실용적으로 반영한다.

### 6.1 Encoder Block 구성

```
1. Layer Normalization
    
2. Multi-Head Self-Attention
    
3. Residual Connection
    
4. Feed Forward Network
    
5. Residual Connection
```

### 6.2 Residual Connection의 역할

- 깊은 네트워크에서도 정보 보존    
- Gradient 흐름 안정화    
- Attention 결과가 원본 표현을 “덮어쓰지 않도록” 보장    


👉 Residual 개념 : (CNN/RNN 공통 기법)

---

## 7. Feed Forward Network를 Conv1D로 구현한 이유

Transformer 논문의 FFN은 Dense 구조이지만, 실습에서는 다음과 같이 변형했다.

```yaml
Conv1D(kernel_size=1) ≈ Dense
```

### 장점

- 위치 독립적 연산 (Dense와 동일)    
- GPU 병렬 처리 효율 ↑    
- Dropout 삽입 용이    
- 실전 코드에서 널리 사용되는 패턴    

> 핵심은 “Conv냐 Dense냐”가 아니라
> **Attention 뒤에서 비선형 변환을 수행한다는 구조적 역할**

---

## 8. Encoder-only Transformer의 활용 (문장 분류)

Transformer는 반드시 Seq2Seq 구조일 필요가 없다.

### 문제 유형별 구조 선택

|**문제 유형**|**구조**|
|---|---|
|번역 / 요약 / 생성|Encoder + Decoder|
|감정 분석 / 분류|Encoder only|

tformer03 실습에서는:
- GlobalAveragePooling으로
    - 시퀀스 전체를 하나의 벡터로 요약
    
- RNN의 “마지막 hidden state” 개념을 대체    


👉 Seq2Seq 구조
\
👉 [Seq2Seq 이론 정리](251222_seq2seq.md)

---

## 9. Transformer가 Seq2Seq를 대체한 핵심 이유 (요약)

- RNN 기반 Seq2Seq    
    - 순차 처리        
    - 긴 문장에 취약        
    
- Transformer 기반 Seq2Seq    
    - Attention만으로 인코딩        
    - 모든 위치 직접 참조        
    - 병렬 처리 가능        

> **“Attention is All You Need”의 의미는**
> **Attention만으로도 시퀀스 표현이 충분하다는 구조적 증명**

---

## 🔎 전체 요약

> Transformer는 시간 순서가 아닌 **관계 중심 모델**로,
> Self-Attention과 Multi-Head 구조를 통해
> **RNN 없이도 문맥/의미/장기 의존성을 동시에 해결한 시퀀스 모델**이다.