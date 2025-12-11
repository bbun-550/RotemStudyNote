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
│── rag06_pdf.ipynb
│── rag07_pdf_table.ipynb
│── rag08_pdf_img.ipynb
│── rag09_pdf_overlap.ipynb
│── rag10_routing.ipynb
│── rag11_async.ipynb
│── streamlit_ex.py
└── README.md
```


## 🔍 실습 파일별 개요

⚠️ 학습 진행중 ...


### 📌 rag01.ipynb — 이미지 기반 설명 생성 & 멀티모달 생성 실습

개요:
Gemini 멀티모달 모델을 사용해 이미지 설명 생성과 이미지 기반 시(詩) 생성을 수행하는 실습.
RAG의 이미지 문서 처리 파이프라인을 이해하기 위한 기반 Notebook.

핵심 내용:

	•	Google Gemini 멀티모달 API 연동
	•	시스템 프롬프트 + 이미지 + 사용자 요청 구조 실습
	•	이미지 → 텍스트 변환(Image Captioning)
	•	클래스 기반 멀티모달 생성기 구현
	•	스트리밍 출력 구조 이해

이론 요약:

	•	멀티모달 모델 개념
	•	이미지 설명 생성(Image Captioning)의 RAG 활용성
	•	시스템 프롬프트 역할
	•	stream=True/False 차이

rag01.ipynb - 실습의  의미

	•	아직 검색(Retrieval)을 다루지 않지만 멀티모달 입력을 다루는 첫 실습으로서 
	이미지 → 텍스트 변환이라는 RAG 입력 확장 기반을 제공
	•	이후 PDF 이미지 추출(rag08, rag09)과도 연계됨

👉 [RAG & 멀티모달 기본 이론 정리](90.course_notes/00.NOTES/251203_rag.md)￼

---

### 📌 rag02.ipynb — 텍스트 기반 End-to-End RAG 파이프라인

개요:

텍스트 데이터로 가장 기본적인 RAG(Retrieval-Augmentation-Generation) 파이프라인을 직접 구현
문서 로딩 → 텍스트 분리 → 임베딩 생성 → VectorDB 저장 → 검색(R) → 프롬프트 증강(A) → 최종 생성(G)까지 RAG의 모든 단계를 코드 레벨에서 체득

특히 다음 요소를 중점적으로 다룬다:

	•	LangChain 없이 직접 구현하는 RAG 핵심 구조 이해
	•	SentenceTransformer(MiniLM) 기반 텍스트 임베딩
	•	ChromaDB를 사용한 VectorDB 구축 및 검색
	•	검색된 문서를 기반으로 프롬프트를 강화(Augmented Prompt)
	•	Gemini를 활용한 최종 답변 생성

핵심 내용 요약:
1. 텍스트 기반 RAG 파이프라인 구조

RAG는 크게 3단계로 이루어진다:

① Retrieval — 외부 문서에서 필요한 정보 검색

	•	foods.txt 로딩 → 문장 단위로 청크 생성
	•	SentenceTransformer로 임베딩 생성
	•	ChromaDB에 저장
	•	query 임베딩 → cosine distance 기반 유사 문서 검색

Retrieval 단계 = “LLM이 모르는 정보”를 외부에서 찾아오는 단계

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

	•	예시 제공
	•	배경지식 포함
	•	마크업 금지
	•	친절한 말투 유지

즉, Retrieval 결과 기반 답변 스타일 제어에 초점을 둔다.

4. End-to-End RAG 흐름 심화
`저장된 문서 직접 검증 → 검색 → context 구성 → 스타일이 있는 Augmented Prompt → LLM 생성`
실전 RAG 파이프라인에 가까운 형태



### 📌 rag04_web.ipynb — (Web Crawling + RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 rag05_web_realtime.ipynb — (실시간 웹 검색 RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 rag06_pdf.ipynb — (기본 PDF RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 rag07_pdf_table.ipynb — (PDF Table RAG)

개요:

핵심 내용:
	•	
	•	


### 📌 rag08_pdf_img.ipynb — (PDF Image Extraction + RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 rag09_pdf_overlap.ipynb — (Chunk Overlap 실험)

개요:

핵심 내용:
	•	
	•	



### 📌 rag10_routing.ipynb — (Router / Branching RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 rag11_async.ipynb — (비동기 RAG / Async RAG)

개요:

핵심 내용:
	•	
	•	



### 📌 streamlit_ex.py — (Streamlit 기반 RAG Demo UI)

개요:


핵심 기능:
	•	
	•	