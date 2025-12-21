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

👉 [RAG & 멀티모달 기본 이론 정리](00.NOTES/251203_rag.md)￼

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

👉 [텍스트 기반 RAG 파이프라인 이론 정리](00.NOTES/251203_rag.md)￼

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

👉 [VectorDB 내부 조회 & 고급 프롬프트 엔지니어링 이론 정리](00.NOTES/251209_rag.md)￼

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

### 📌 rag06_pdf.ipynb — PDF 문서 로딩과 페이지 단위 요약

개요: 

PDF 문서를 RAG의 지식 소스를 사용하기 위해 **페이지 단위 텍스트 추출 및 LLM 요약** 수행

핵심 내용:
- PyPDFLoader를 이용한 페이지 단위 문서 로딩
- `Document.page_content, metadata` 구조 이해
- LLM을 활용한 
  - 페이지별 요약
  - 전체 문서 요약
- 문서 요약을 **Retrieval 이전 전처리 단계**로 활용하는 개념

---

### 📌 rag07_pdf_table.ipynb — PDF 표(Table) 추출 및 구조화 실습

개요:

**표 형식 데이터가 포함된 PDF 문서**를 대상으로  
단순 텍스트 추출을 넘어 **표(Table) 데이터를 구조적으로 추출, 정제**하는 과정을 다룬다.

`PyMuPDFLoader`를 이용한 **전체 텍스트 추출**과  
`pdfplumber`를 이용한 **표 전용 추출 방식**을 비교·활용하여,

- 비정형 PDF 문서에서
- 정형 데이터(DataFrame)를 생성하고
- CSV / Excel 파일로 저장하는 흐름

을 실습한다.

핵심 내용:
- **PDF 전체 텍스트 추출 (PyMuPDFLoader)**
  - 페이지 단위로 PDF 로딩
  - `Document.page_content`를 이용한 전체 문자열 결합
  - 정규표현식을 활용한 **한글 텍스트만 추출**
  - 여러 줄 텍스트를 하나의 긴 문장으로 정제

- **표(Table) 데이터 전용 추출 (pdfplumber)**
  - `page.extract_tables()`를 이용한 표 리스트 추출
  - 헤더 행 제거 후 데이터 행만 처리
  - 열(column) 개수 검증을 통한 안정적 파싱
  - 분리된 셀을 병합하여 명칭 컬럼 재구성

- **구조화된 데이터 생성**
  - 시도 / 시군구 / 명칭 / 유형 / 페이지 번호를 컬럼으로 구성
  - 각 행(row)을 하나의 딕셔너리로 관리
  - Pandas DataFrame으로 변환

- **기초 통계 분석**
  - 유형(계곡, 하천 등)별 빈도수 집계
  - PDF 문서를 데이터 분석 관점에서 재해석

- **데이터 파일로 저장**
  - CSV 파일 저장
  - Excel 파일 저장
  - 이후 RAG, 통계 분석, 시각화 실습에 재사용 가능

---

### 📌 rag08_pdf_img.ipynb — PDF 이미지(OCR) 기반 멀티모달 RAG 전처리

개요:

**PDF 문서에 포함된 이미지 영역**(스캔 문서, 사진 등)을 대상으로  
**OCR(Optical Character Recognition)** 을 적용하여 텍스트를 추출하는 과정을 다룬다.

- **텍스트가 아닌 이미지 형태로 존재하는 정보까지 RAG에 포함**시키는 단계이다.

이를 통해 **멀티모달 문서 처리 기반 RAG의 기초 전처리 파이프라인**을 학습한다.


핵심 내용:

- **PDF 텍스트 추출 방식 비교**
  - `PyMuPDFLoader`를 이용한 페이지 단위 텍스트 로딩
  - `fitz(PyMuPDF)`를 이용한 저수준 PDF 접근
  - `Document(page_content, metadata)` 구조 직접 생성

- **PDF 메타데이터 활용**
  - 페이지 번호, 전체 페이지 수
  - 작성자(author), 제목(title), 생성 도구(producer)
  - 추후 RAG 답변의 출처·신뢰도 설명에 활용 가능

- **PDF 이미지 추출**
  - `page.get_images(full=True)`를 이용해 페이지 내 모든 이미지 탐색
  - `doc.extract_image(xref)`로 이미지 raw bytes 추출
  - PIL.Image로 변환 후 파일 저장

- **OCR(광학 문자 인식) 적용**
  - `pytesseract.image_to_data()`를 사용하여
    - 텍스트
    - 위치 좌표
    - 신뢰도(confidence)
    를 함께 추출
  - 신뢰도(conf) 60 이상 단어만 필터링하여 노이즈 제거

- **OCR 결과 시각화**
  - 인식된 텍스트 영역에 사각형(box) 표시
  - OCR 결과를 이미지 위에 직접 확인
  - OCR 품질 검증 및 디버깅에 활용

- **한글 OCR 환경 구성**
  - `tesseract-ocr-kor` 설치
  - 한국어 + 영어(`kor+eng`) 동시 인식 설정

- **OCR 정확도 개선 기법**
  - 이미지 확대(resize) 후 OCR 수행
  - 정규표현식으로
    - 한글/영어
    - 2글자 이상
    만 추출하여 의미 없는 결과 제거

- **OCR 결과 구조화**
  - 페이지 번호 + 이미지 번호 + 추출 텍스트 형태로 저장
  - 이후 VectorDB 저장 및 RAG Retrieval 대상 확장 가능

---

### 📌 rag09_pdf_overlap.ipynb — Chunk Overlap을 통한 문맥 보존 전략

개요:

RAG 시스템에서 PDF나 긴 문서를 그대로 **페이지 단위 또는 고정 청크 단위**로 분할하면,  
문맥이 중요한 문장들이 경계에서 끊어지는 문제가 발생한다.

예를 들어,
- 이전 페이지: *“그는 아내를 찾아…”*
- 다음 페이지: *“간신히 병원에 도착했다.”*

처럼 나뉘면,
- Retrieval 단계에서 한 쪽만 검색될 경우
- **주어, 행위, 맥락이 소실된 채 LLM에 전달**된다.

이 문제를 해결하기 위해 
**이전 페이지(또는 이전 청크)의 일부 문단을 다음 페이지 앞에 겹쳐 붙이는  
Overlap 기반 문서 분할 전략**을 직접 구현하고 검증한다.


핵심 내용:

- **문맥 단절 문제 인식**
  - 페이지 단위 분리 시 질문 응답 정확도 저하
  - Retrieval 결과가 “불완전한 정보 조각”이 되는 문제

- **PDF 로딩 구조 이해**
  - `PyMuPDFLoader`로 PDF 로드
  - 페이지마다 하나의 `Document`
    - `page_content`: 해당 페이지 전체 텍스트
    - `metadata`: 페이지 번호, 전체 페이지 수, 작성자 등

- **문단 단위 분리**
  - 한 페이지 내 텍스트를 `\n` 기준으로 분리
  - 불필요한 공백 제거 후 문단 리스트 생성
  - 전체 문서를 “문단 단위 지식 조각”으로 변환

- **Chunk Overlap(슬라이딩 윈도우) 구현**
  - 이전 페이지의 마지막 N개 문단을
    → 다음 페이지 문단 앞에 **겹쳐서 결합**
  - `overlap_count = 2` 등으로 겹침 범위 조절 가능
  - 페이지 경계를 넘어 자연스러운 문맥 유지

- **Overlap 추적 및 검증**
  - 어떤 페이지의 문단이
    - 어느 페이지로 겹쳐 들어갔는지 기록
  - `[p.이전 → p.다음]` 형태로 로그 출력
  - 실제로 문맥 연결이 유지되는지 확인

- **Overlap 적용 전/후 비교**
  - overlap 미적용: 총 문단 수 ↓, 문맥 단절 발생
  - overlap 적용: 문단 수 ↑, Retrieval 품질 개선
  - “문단 수 증가”는 비용이지만
    → **정확도·이해도 향상이라는 실질적 이득**이 있음
---

### 📌 rag10_routing.ipynb — 질문 유형 분기 기반 멀티체인 RAG(Router)

개요:

`rag10.py`는 `data/` 폴더의 `.txt` 강의 노트를 읽어
1) 문서를 청크로 분할하고
2) SentenceTransformer로 임베딩하여
3) Chroma 벡터DB에 저장/로드한 뒤
4) 사용자의 질문을 **RAG / 코드헬프 / 일반설명** 중 하나로 라우팅해서
5) 적절한 체인으로 답변하는 “튜터형 챗봇”을 만든다

핵심은 “강의자료 기반 답변(RAG)”을 기본으로 하되,
질문 성격에 따라 **디버깅 전용 프롬프트**나 **일반 설명 프롬프트**로 분기하는 구조까지 포함한다.

핵심 내용:
1) 환경/경로 구성
2) SentenceTransformer 임베딩을 LangChain Embeddings로 래핑
3) 문서 로딩 및 청킹(Chunking)
4) Chroma 벡터DB 생성/재사용
5) RAG 체인(LCEL) 구성
6) RAG 없이 사용하는 일반 체인 / 코드 헬프 체인
7) 질의 라우팅(에이전트 브레인)
8) 인터랙티브 채팅 루프 + 참고 chunk 출력

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

### 📌 rag14_conditional_edge.ipynb — 

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

---

### 📌 rag16_llm_graph.ipynb — LangGraph 기반 LLM 파이프라인 기초

개요:

LangGraph를 활용해 LLM 기반 작업 흐름을 그래프(Graph) 구조로 구성하는 첫 실습이다.
기존의 “한 번 호출 → 한 번 응답” 방식에서 벗어나,
- 질문 처리
- LLM 답변 생성
- 후처리(요약)

와 같은 단계를 명시적인 노드(Node)와 엣지(Edge) 로 분리하여
LLM 애플리케이션을 구조적·확장 가능하게 설계하는 방법을 학습한다.

RAG를 직접 사용하지는 않지만,
\
➜ 이후 rag17의 RAG + Graph 결합 구조를 이해하기 위한 필수 기초 단계에 해당한다.

핵심 내용:

1. State 기반 LLM 설계 (TypedDict)

```python
class QAState(TypedDict):
    question: str
    answer: str
    summary: str
```

- LangGraph는 State 중심(Stateful) 설계를 사용
- 각 노드는 State를 입력으로 받고, State를 업데이트하여 반환
- 단순 문자열 전달이 아니라 파이프라인 전체의 데이터 흐름을 명시적으로 관리

2. LLM 처리 노드 분리

- 답변 생성 노드 — `node_llm_answer`
	- 역할: 질문 → LLM 호출 → 답변 생성
	- 프롬프트 책임:
	- 한국어 응답
	- 중학생 눈높이 설명
	- 10문장 이내 제약

Question → LLM → Answer
\
➜ LLM 호출을 하나의 독립 노드로 캡슐화하는 것이 핵심

- 후처리 노드 — `node_summarize`
	- 역할: LLM 답변 결과를 요약 형태로 가공
	- 실제 LLM 요약 대신:
	- 문자열 전처리
	- 길이 제한
	- 요약 포맷 생성

> Answer → Post-Processing → Summary

➜ LLM을 쓰지 않는 후처리도 Graph 노드로 분리 가능하다는 점이 중요

3.  Graph 구조 정의 (StateGraph)

`[ llm_answer ] → [ summarize ] → END`

- Entry Point: llm_answer
- 순차 실행 구조
- 각 단계의 책임이 명확히 분리됨

```python
graph.add_edge("llm_answer", "summarize")
graph.add_edge("summarize", END)
```

➜ 절차형 코드가 아닌, 실행 흐름을 선언적으로 정의

4. Graph 실행 & 결과 State 확인

```python
final_state = app.invoke(init_state)
```

- 실행 결과는 단순 출력이 아니라 최종 State
- 하나의 실행에서:
- question
- answer
- summary
를 모두 보존

➜ 디버깅, 로그 저장, 재실행에 매우 유리한 구조

5. Graph 시각화 (Mermaid)

```python
g = app.get_graph()
g.draw_mermaid_png()
```

- LangGraph의 가장 강력한 장점 중 하나
- LLM 애플리케이션의 실행 흐름을 시각적으로 검증 가능
- 노드/엣지 구조를 한눈에 파악

이 실습의 의미

구조 : LLM 작업을 Graph로 표현
\
설계 : 상태(State) 기반 파이프라인
\
확장 : 조건 분기, 반복, 병렬 처리의 기반
\
연결 : rag17 (RAG + Graph)로 자연스럽게 확장

➜ rag16은 “LLM을 함수처럼 쓰는 단계”에서 “LLM을 시스템처럼 설계하는 단계”로 넘어가는 분기점이다.

---

### 📌 rag17_chain_graph.ipynb — LangGraph 기반 분기형 RAG + LLM 멀티체인 챗봇

개요:

LangChain과 LangGraph를 결합하여 “질문 유형에 따라 서로 다른 응답 체인으로 분기하는 챗봇”을 구현 실습
- 사내 문서 관련 질문 → VectorDB 기반 RAG 체인
- 일반 상식 질문 → 순수 LLM 체인
- 분류가 애매한 경우 → Fallback 체인

이 전체 흐름을 LangGraph가 워크플로우(Graph)로 제어하고,
각 노드 내부의 실제 LLM / RAG 로직은 LangChain(LCEL) 이 담당한다.

➜ 즉,
“LLM 작업은 LangChain, 전체 판단·분기 흐름은 LangGraph” 라는 역할 분리를 명확히 보여주는 예제이다.


핵심 내용:

1. 사내 문서 기반 RAG 체인 구성
- OpenAI Embedding(text-embedding-3-small)을 사용해 사내 문서 임베딩
- ChromaDB에 벡터 저장 (company-col)
- Retriever(k=3)를 통해 관련 문서 검색
- 검색 결과를 Prompt에 삽입하여 답변 생성

RAG 체인의 역할
- 회사 규정, 근무시간, 연차, 복지 등 → 모델이 “모른다”고 가정해야 할 내부 정보 처리


2. 일반 상식 LLM 체인 분리
- VectorDB / Retrieval 없이
- 단순 질문 → LLM 답변 구조

의미
- 모든 질문에 RAG를 쓰지 않아도 됨
- 불필요한 검색 비용 및 지연 감소
- “RAG는 필요할 때만 사용”이라는 실전 설계 관점 반영


3. 질문 분류(Classification) 체인

질문을 다음 중 하나로 분류:
- "rag" : 사내 정책·회사 규정 관련
- "llm" : 일반 상식
- "unknown" : 애매한 경우

분류 기준을 프롬프트로 명시하고,
LLM이 반드시 "rag" 또는 "llm" 중 하나를 출력하도록 강제한다.

➜ LLM을 ‘판단자(Classifier)’로 사용하는 패턴을 학습


4. LangGraph 기반 조건 분기 워크플로우

Graph 구조 요약:

classify
   ├─ rag  → rag_answer → END
   ├─ llm  → llm_answer → END
   └─ unknown → fallback → END

- Entry Point: classify
- add_conditional_edges를 이용한 동적 분기
- 분기 결과에 따라 서로 다른 체인 실행

➜ 절차형 if/else 코드가 아닌,
선언적인 그래프 기반 흐름 제어


5. Fallback 노드 설계
- 분류 결과가 애매할 경우 대비
- 사용자에게 상황을 설명한 뒤 일반 LLM 답변 제공

실전 의미
- LLM 분류의 불확실성 대비
- 사용자 경험(UX) 안정성 확보
- 프로덕션 환경에서 매우 중요한 방어 로직

이 실습의 의미

**핵심**
아키텍처 : RAG + 일반 LLM을 하나의 시스템으로 통합
\
역할 분리 : LangGraph(흐름) vs LangChain(실행)
\
확장성 : 질문 유형 추가, 체인 추가가 쉬움
\
실전성 : 사내 챗봇·헬프데스크 구조와 거의 동일

➜ “현실적인 멀티 소스 챗봇 아키텍처”로 진입하는 단계

---

### 📌 rag18_workflow.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag19_graph_chkpoint.ipynb — 

개요:

핵심 내용:
	•	
	•	

---

### 📌 rag20_graph_llm.ipynb — 

개요:

핵심 내용:
	•	
	•	

---