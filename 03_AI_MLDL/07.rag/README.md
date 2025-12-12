## 📘 Retrieval-Augmented Generation (RAG) 실습 정리

이 폴더는 RAG 파이프라인 학습을 위한 실습 및 정리 파일을 모아둔 공간입니다.
웹 기반 RAG, PDF 기반 RAG, Table/이미지 혼합 문서, Streaming, Async, Routing 등 다양한 형태의 Retrieval 실험을 다룹니다.


## 📂 Folder Structure

```yaml
07.rag/
│── rag01.ipynb
│── rag02.ipynb
│── rag03.ipynb
│── rag04_web.ipynb
│── rag05_web_realtime.ipynb
│── streamlit_ex.py
│── rag06_pdf.ipynb
│── rag07_pdf_table.ipynb
│── rag08_pdf_img.ipynb
│── rag09_pdf_overlap.ipynb
│── rag10_routing.ipynb
│── rag11_async.ipynb
│── rag12_langgraph.ipynb
│── rag13_langgraph.ipynb
│── rag14_conditional_edge.ipynb
│── rag15_conditional_edge.ipynb
└── README.md
```


## 🔍 실습 파일별 개요

⚠️ 학습 진행중 ...


### 📌 rag01.ipynb — 이미지 기반 설명 생성 & 멀티모달 생성 실습

개요:
Gemini 멀티모달 모델을 사용해 이미지 설명 생성과 이미지 기반 시(詩) 생성을 수행하는 실습.
RAG의 이미지 문서 처리 파이프라인을 이해하기 위한 기반 Notebook.

핵심 내용:

- Google Gemini 멀티모달 API 연동
- 시스템 프롬프트 + 이미지 + 사용자 요청 구조 실습
- 이미지 → 텍스트 변환(Image Captioning)
- 클래스 기반 멀티모달 생성기 구현
- 스트리밍 출력 구조 이해

이론 요약:

- 멀티모달 모델 개념
- 이미지 설명 생성(Image Captioning)의 RAG 활용성
- 시스템 프롬프트 역할
- stream=True/False 차이

rag01.ipynb - 실습의  의미

- 아직 검색(Retrieval)을 다루지 않지만 멀티모달 입력을 다루는 첫 실습으로서 
	이미지 → 텍스트 변환이라는 RAG 입력 확장 기반을 제공
- 이후 PDF 이미지 추출(rag08, rag09)과도 연계됨

👉 [RAG & 멀티모달 기본 이론 정리](90.course_notes/00.NOTES/251203_rag.md)￼

---

### 📌 rag02.ipynb — 텍스트 기반 End-to-End RAG 파이프라인

개요:

텍스트 데이터로 가장 기본적인 RAG(Retrieval-Augmentation-Generation) 파이프라인을 직접 구현
문서 로딩 → 텍스트 분리 → 임베딩 생성 → VectorDB 저장 → 검색(R) → 프롬프트 증강(A) → 최종 생성(G)까지 RAG의 모든 단계를 코드 레벨에서 체득

특히 다음 요소를 중점적으로 다룬다:

- LangChain 없이 직접 구현하는 RAG 핵심 구조 이해
- SentenceTransformer(MiniLM) 기반 텍스트 임베딩
- ChromaDB를 사용한 VectorDB 구축 및 검색
- 검색된 문서를 기반으로 프롬프트를 강화(Augmented Prompt)
- Gemini를 활용한 최종 답변 생성

핵심 내용 요약:
1. 텍스트 기반 RAG 파이프라인 구조

RAG는 크게 3단계로 이루어진다:

① Retrieval — 외부 문서에서 필요한 정보 검색

- foods.txt 로딩 → 문장 단위로 청크 생성
- SentenceTransformer로 임베딩 생성
- ChromaDB에 저장
- query 임베딩 → cosine distance 기반 유사 문서 검색

Retrieval 단계 = “LLM이 모르는 정보”를 외부에서 찾아오는 단계

👉 [텍스트 기반 RAG 파이프라인 이론 정리](90.course_notes/00.NOTES/251205_rag.md)￼

---

### 📌 rag03.ipynb — VectorDB 조회/ID 검색 + 고급 프롬프트 엔지니어링(텍스트 기반 RAG 심화)

개요:

rag03 실습은 rag02.ipynb에서 구현한 기본 텍스트 기반 RAG 구조를 확장하여
**VectorDB 내부 조회 기능(get, count, id 기반 검색)**과
검색된 문서를 활용하는 고급 프롬프트 엔지니어링 기법을 학습

주요 학습 포인트
	1.	ChromaDB 내부 구조를 이해하고, ID 기반 문서 조회 → VectorDB를 검증하는 방법을 학습한다.
	2.	Retrieval 결과를 활용해 “예시 포함, 배경지식 추가, 마크다운 금지”와 같은 응답 스타일을 정교하게 제어하는 Prompt Engineering 기술을 다룬다.

핵심 내용 요약:

1.  VectorDB 내부 구조 조회
	•	`collection.get()`  → 저장된 문서/임베딩/메타데이터 전체 확인
	•	`collection.count()` → VectorDB 문서 수 확인
	•	`collection.get(ids=["doc_0"])` → 특정 ID 문서를 직접 조회

이를 통해 실제 VectorDB가 어떤 형태로 데이터를 저장하는지 학습

2. 검색(Retrieval)

rag02와 동일하게 query 벡터 생성 후 유사 문서 검색하지만 rag03.ipynb에서는 아래와 같이 다룸

```python
context = "\n".join(results["documents"][0])
```
documents가 2차원 배열로 제공되는 구조를 이해하는 것이 핵심

3. 고급 Prompt Engineering

LLM 프롬프트

- 예시 제공
- 배경지식 포함
- 마크업 금지
- 친절한 말투 유지

즉, Retrieval 결과 기반 답변 스타일 제어에 초점을 둔다.

4. End-to-End RAG 흐름 심화
`저장된 문서 직접 검증 → 검색 → context 구성 → 스타일이 있는 Augmented Prompt → LLM 생성`
실전 RAG 파이프라인에 가까운 형태

👉 [VectorDB 내부 조회 & 고급 프롬프트 엔지니어링 이론 정리](90.course_notes/00.NOTES/251209_rag.md)￼

---

### 📌 rag04_web.ipynb — (Web Crawling + RAG)

개요:

웹 문서를 직접 크롤링하여 Retrieval-Augmented Generation(RAG)을 구성하는 전체 파이프라인 구현  
> Wikipedia 외부 웹 페이지를 데이터 소스로 활용

- 웹 페이지 수집
- 텍스트 전처리
- 문서 임베딩
- 벡터 데이터베이스(ChromaDB) 저장
- 유사 문서 검색
- 검색 결과를 활용한 LLM 응답 생성

> **RAG 기본 구조를 실습 중심으로 학습**
\
> **외부 지식을 기반으로 답변을 생성하는 방식**임을 이해하는 것이 목표이다.

핵심 내용:
- **웹 크롤링 기반 데이터 수집**
  - `requests` + `BeautifulSoup`를 활용하여 웹 페이지 HTML 요청
  - `<p>` 태그 기반 텍스트 추출
  - User-Agent 설정을 통한 스크래핑 안정성 확보

- **SentenceTransformer를 활용한 문서 임베딩**
  - `all-MiniLM-L6-v2` 모델 사용
  - 문단 단위 텍스트를 고정 차원 벡터(384차원)로 변환

- **ChromaDB 벡터 데이터베이스 구축**
  - `PersistentClient`를 사용하여 로컬 디스크 기반 벡터 저장
  - 문서(id, embedding, text) 단위 저장 및 관리
  - 저장된 벡터 수 및 샘플 문서 검증

- **질의 임베딩 및 유사 문서 검색**
  - 사용자 질문을 동일한 임베딩 공간으로 변환
  - cosine distance 기반 Top-k 문서 검색
  - 검색 결과의 거리(distance)를 통해 유사도 확인

- **검색 결과 기반 Prompt 강화**
  - 검색된 문서를 Prompt에 직접 삽입
  - LLM이 외부 지식을 참조하도록 유도
  - 답변 길이, 형식, 내용 범위를 명시적으로 제어

- **LLM(Gemini) 기반 응답 생성**
  - 검색 결과에 한정된 정보만 사용하도록 프롬프트 설계
  - RAG 구조에서 Retrieval과 Generation의 역할 분리 이해

---

### 📌 rag05_web_realtime.ipynb — (실시간 웹 검색 RAG)

개요:

**실시간 웹 검색 결과를 활용하는 Realtime RAG(Web RAG)** 를 구현

> **질문 시점에 웹 검색 → 결과 요약 → 답변 생성**이라는 흐름
- 최신 정보 반영
- 외부 지식 의존성 최소화
- 검색과 생성의 역할 분리

> **실전형 RAG 아키텍처**를 학습하는 것이 목표이다.

본 실습에서는 Tavily Search API와 LLM(Gemini)을 결합하여  
**검색 → 요약 → 답변 생성의 2단계 RAG 파이프라인**을 클래스 기반으로 구현한다.


핵심 내용:
- **Realtime Web Search 기반 Retrieval**
  - `TavilySearch`를 활용한 실시간 웹 검색 수행
  - 질문 입력 시마다 최신 검색 결과를 동적으로 수집
  - 정적 문서 기반 RAG의 한계를 보완

- **검색 결과 요약(Summarization) 단계 분리**
  - 검색 결과를 그대로 LLM에 전달하지 않고, 1차 요약 수행
  - 광고/중복/홍보성 콘텐츠 제거
  - 핵심 정보만 압축하여 Answer 단계로 전달

- **Prompt 역할 분리 설계**
  - 검색 결과 요약 전용 프롬프트
  - 최종 답변 생성 전용 프롬프트
  - 각 단계별 LLM 책임 범위를 명확히 분리

- **LLM 기반 Answer Generation**
  - 요약된 검색 결과에 한정하여 답변 생성
  - 추측·환각(hallucination) 방지를 위한 제약 조건 포함
  - “모르는 내용은 모른다고 답변”하도록 프롬프트 설계

- **클래스 기반 RAG 파이프라인 구성**
  - `OptimizeWebRAG` 클래스로 전체 흐름 캡슐화
  - 검색 → 요약 → 답변 과정을 메서드 단위로 분리
  - 재사용성과 확장성을 고려한 구조

- **API Rate Limit 고려**
  - 연속 호출 제한을 고려한 `sleep()` 적용
  - 실시간 검색 기반 RAG에서 발생할 수 있는 호출 제한 이슈 인지
---

### 📌 streamlit_ex.py — (Streamlit 기반 RAG UI)

개요:

**VectorDB 기반 RAG 파이프라인**을 **Streamlit 웹 인터페이스와 결합하여**, 사용자가 직접 질문을 입력하고 LLM의 응답을 확인할 수 있는  
**인터랙티브 RAG 애플리케이션** 구현

SentenceTransformer를 활용한 임베딩, ChromaDB를 이용한 벡터 검색, Gemini LLM을 통한 답변 생성을 하나의 웹 UI 흐름으로 연결함으로써  
**RAG의 전체 구조를 시각적으로 이해 목표**

핵심 기능:
- **RAG 파이프라인 UI 통합**
  - 기존 콘솔 기반 RAG 로직을 Streamlit UI로 확장
  - 사용자의 질문 입력 → 검색 → 답변 생성까지 실시간 처리

- **VectorDB 기반 Retrieval 유지**
  - `SentenceTransformer(all-MiniLM-L6-v2)`로 문서 및 질문 임베딩
  - ChromaDB에 사전 저장된 지식 데이터 검색
  - 질문과 의미적으로 가장 유사한 문서 Top-K 조회

- **Streamlit 입력/출력 구성**
  - `st.text_input()`으로 사용자 질문 입력
  - 질문이 입력될 때만 RAG 로직 실행
  - LLM 응답을 `st.write()`로 출력

- **LLM 응답 생성 흐름**
  - 검색된 문서를 하나의 Context로 결합
  - Context + Question 기반 프롬프트 구성
  - Gemini 모델을 이용해 최종 답변 생성

- **UX 개선 요소**
  - `st.spinner()`를 활용한 응답 생성 대기 표시
  - LLM 처리 중 사용자 피드백 제공
  - 콘솔 출력 없이 웹 기반 인터랙션 가능

---

### 📌 rag06_pdf.ipynb — (기본 PDF RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag07_pdf_table.ipynb — (PDF Table RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag08_pdf_img.ipynb — (PDF Image Extraction + RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag09_pdf_overlap.ipynb — (Chunk Overlap 실험)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag10_routing.ipynb — (Router / Branching RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag11_async.ipynb — (비동기 RAG / Async RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag11_async.ipynb — (비동기 RAG / Async RAG)

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag12_langgraph.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag13_langgraph.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag14_conditional_edge.ipyn — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag15_conditional_edge.ipynb — 

개요:

핵심 내용:
	•	
	•	

