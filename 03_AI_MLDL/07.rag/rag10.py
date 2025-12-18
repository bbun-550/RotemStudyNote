import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 환경 설정 =====
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / ".chroma_tutor_db"

load_dotenv()

# SentenceTransformer → LangChain Embeddings 래핑 =====
class SentenceTransformerEmbeddings(Embeddings):
    # SentenceTransformer를 LangChain Embeddings 인터페이스에 맞게 감싸는 래퍼.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 여러 개의 텍스트(문서 chunk 리스트)를 한 번에 임베딩 후, 파이썬 리스트로 변환
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # 하나의 질의문을 임베딩 (model.encode는 리스트 입력을 기대하므로 [text] 형태로 전달)
        return self.model.encode([text])[0].tolist()
    
# 문서 로딩 & Chroma 벡터DB 구축 =====
def load_text_documents(data_dir: Path):
    # data_dir 아래의 모든 .txt 파일을 LangChain Document 리스트로 로드.
    docs = []
    for path in data_dir.glob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        # loader.load()는 List[Document] 를 반환하므로 docs 리스트에 계속 이어 붙임
        docs.extend(loader.load())
    if not docs:
        # 학습/검색에 사용할 문서가 하나도 없으면 바로 예외 발생시켜 초기 세팅 단계에서 문제 인지
        raise RuntimeError(f"{data_dir} 폴더에 .txt 문서가 없습니다.")
    return docs


def split_documents(documents, chunk_size: int = 300, chunk_overlap: int = 50):
    # 긴 문서를 chunk 단위로 나누기. (검색 정밀도를 위해 300/50으로 조정)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, # 앞/뒤 chunk간 겹치는 문자 수 → 문맥 유지에 도움
        separators=["\n\n", "\n", " ", ""],
        # 우선적으로 나눌 구분자 우선순위 (단락 → 줄 → 공백 → 기타)
    )
    # List[Document] → 더 잘게 쪼개진 List[Document]
    return splitter.split_documents(documents)


def get_or_create_vectorstore():
    """
    - 이미 Chroma DB가 있으면 그대로 로드
    - 없으면 data/ 폴더의 문서를 읽어서 새로 구축
    """
    # Chroma에 넘길 임베딩 함수로 SentenceTransformer 래퍼 사용
    embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

    if CHROMA_DIR.exists():
        print("[INFO] 기존 Chroma DB 로딩...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),   # 디스크에 저장된 Chroma 인덱스 경로
            embedding_function=embedding_model,  # 질의/문서 임베딩에 사용할 함수
        )
    else:
        # 최초 실행 시: data 폴더에서 txt 문서를 읽어와 새로 벡터DB 생성
        print("[INFO] 새 Chroma DB 생성 중...")
        raw_docs = load_text_documents(DATA_DIR)   # 원문 Document 리스트 로드
        chunked_docs = split_documents(raw_docs)   # 검색용으로 chunk 분할

        vectorstore = Chroma.from_documents(
            documents=chunked_docs,     # 분할된 Document들을 바로 저장
            embedding=embedding_model,  # from_documents에서는 embedding 파라미터 사용
            persist_directory=str(CHROMA_DIR), # 생성된 인덱스를 디스크에 영구 저장
        )
        print(f"[INFO] 문서 {len(chunked_docs)}개를 벡터화하여 Chroma에 저장 완료.")
    return vectorstore


# LLM(Gemini) & RAG 체인 정의 =====
def build_llm():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    return llm

def build_rag_chain(vectorstore, llm):
    """
    LangChain LCEL 스타일의 RAG 체인.
    - 입력: 사용자의 질문(str), 출력: 최종 답변(str)
    """

    # RAG에서 사용할 Retriever 생성 (k=5 → 상위 5개 chunk 검색)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Document 리스트를 하나의 문자열 context로 합치는 함수
    def _format_docs(docs):
        if not docs:
            # 검색 결과가 하나도 없을 때는 context가 비어있지 않도록 안전 장치
            return "[출처 없음]\n(관련 문서를 찾지 못했습니다.)"
        # 각 Document에 대해 [source: 파일명] 헤더를 붙이고 본문(page_content)을 이어붙임
        return "\n\n".join(
            f"[source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n"
            f"{d.page_content}"
            for d in docs
        )

    # LCEL에서 파이썬 함수를 Runnable로 감싸기 위해 RunnableLambda 사용
    format_docs = RunnableLambda(_format_docs)

    # 프롬프트: role + 제약사항 + 질문/컨텍스트 슬롯 정의
    prompt = ChatPromptTemplate.from_template("""
      당신은 "데이터 분석/AI 커리큘럼 튜터"입니다.
      아래의 참고 자료(context)만을 기반으로, 질문에 친절하게 답변해 주세요.

      요구사항:
      - 초보자도 이해할 수 있도록 단계별로 설명합니다.
      - 필요하면 간단한 코드 예제를 포함합니다.
      - 자료에 없는 내용은 "자료에 없는 내용이라 정확히 답변하기 어렵다"고 말합니다.
      - 마지막에 이해 체크용 질문 1~2개를 던져 주세요.
      - 답변에는 마크다운 문법(*, -, **, ``` 등)을 절대 사용하지 않는다.
      - 리스트는 '1)', '2)' 또는 '-' 대신 '•' 문자로 표현한다.
      - 코드 예제는 들여쓰기만 사용하며, ```python 같은 마크다운 코드블록을 사용하면 안 된다.

      질문:
      {question}

      참고 자료 (강의 노트/교재 일부):
      {context}
    """.strip())

    # LCEL 체인
    # 입력: question(str) 하나만 받도록 설계
    rag_chain = (
        {
            # "context" 키에는: question → retriever → 관련 docs → 문자열로 포맷
            "context": retriever | format_docs,
            # "question" 키에는: 사용자의 원 질문을 그대로 전달
            "question": RunnablePassthrough(),
        }
        # 위에서 만든 딕셔너리를 prompt에 채워 넣음
        | prompt
        # 프롬프트 결과를 LLM에 전달해 응답 생성
        | llm
        # ChatMessage 객체 대신 순수 문자열만 꺼내기
        | StrOutputParser()
    )

    # rag_chain: question(str) → answer(str)
    # retriever: question(str) → List[Document]
    return rag_chain, retriever


# 일반 LLM 체인 (RAG 없이) =====
def build_general_chain(llm):
    """
    강의자료에 직접 의존하지 않아도 되는, 일반적인 설명/잡담 등에 사용하는 체인.
    question 문자열 하나만 입력으로 사용.
    """
    prompt = ChatPromptTemplate.from_template("""
        당신은 데이터 분석과 프로그래밍 전반을 도와주는 선생님이다.

      요구사항:
      - 초보자도 이해할 수 있도록 친절하고 단계적으로 설명한다.
      - 필요하면 간단한 예시나 비유를 사용한다.
      - 답변에는 마크다운 문법(*, -, **, ``` 등)을 절대 사용하지 않는다.
      - 리스트는 '1)', '2)' 또는 '•' 문자를 사용한다.
      - 코드 예제가 필요하면 들여쓰기만 사용하고, ```python 같은 마크다운 코드블록은 사용하지 않는다.

      질문:
      {question}
    """.strip())

    # question(str) → {"question": question} → prompt → llm → str
    chain = (
        {"question": RunnablePassthrough()}  # LCEL에서 입력 문자열을 딕셔너리 형태로 매핑
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 코드 헬프/디버깅 체인 =====
def build_code_chain(llm):
    """
    코드 오류, 디버깅, 리팩토링, 구현 방법 질문에 특화된 체인.
    question 문자열 하나만 입력으로 사용.
    """
    prompt = ChatPromptTemplate.from_template("""
      당신은 파이썬/데이터분석/머신러닝 코드를 도와주는 10년차 시니어 개발자다.

      요구사항:
      - 질문에 포함된 코드나 에러 메시지를 먼저 천천히 읽고, 어떤 문제가 있는지 단계별로 분석한다.
      - 가능한 원인을 몇 가지로 나누어 설명하고, 각각에 대해 해결 방법을 제안한다.
      - 필요하면 수정된 코드 예제를 보여준다.
      - 답변에는 마크다운 문법(*, -, **, ``` 등)을 절대 사용하지 않는다.
      - 리스트는 '1)', '2)' 또는 '•' 문자를 사용한다.
      - 코드 예제는 들여쓰기만 사용하고, ```python 같은 마크다운 코드블록을 사용하지 않는다.
      - 너무 많은 부분을 한 번에 고치지 말고, 중요한 부분 위주로 설명한다.

      질문(코드/에러 포함 가능):
      {question}
    """.strip())

    # 일반 체인과 동일한 패턴이지만, 프롬프트만 코드 헬프용으로 특화
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ===== 질의 라우터 체인 (에이전트 브레인) =====
def build_question_router(llm):
    """
    질문을 rag / code / general 셋 중 하나로 분류하는 라우터.
    1차: 규칙 기반 (키워드)
    2차: 필요한 경우에만 LLM 분류 사용
    """

    # 1) 규칙 기반 라우터
    def rule_based_router(question: str) -> str:
        q = question.lower()

        # 코드 관련 키워드: 에러/오류/traceback 등
        code_keywords = [
            "error", "traceback", "exception",
            "오류", "에러", "코드", "debug", "디버깅"
        ]

        # 강의/과목 관련 키워드: 파이썬, 자료형, 판다스, 넘파이, 머신러닝 등
        rag_keywords = [
            "python", "파이썬",
            "자료형", "데이터 타입", "datatype", "data type",
            "pandas", "판다스",
            "numpy", "넘파이",
            "머신러닝", "machine learning", "ml",
            "딥러닝", "deep learning",
            "lecture", "강의", "교재", "노트", "수업"
        ]

        # 1순위: 코드 관련이면 code 체인으로 보내기
        if any(k in q for k in code_keywords):
            return "code"

        # 2순위: 강의/과목/자료형 질문이라면 강제로 RAG 사용
        if any(k in q for k in rag_keywords):
            return "rag"

        return "llm"  # 위에 해당되지 않으면, 최종 판단은 LLM에게 위임

    # 2) LLM 기반 보조 라우터 (rule_based_router가 'llm' 반환한 경우에만 사용)
    router_prompt = ChatPromptTemplate.from_template("""
      당신은 사용자의 질문을 아래 세 가지 타입 중 하나로 분류하는 분류기다.

      - 'rag'     : 우리 강의자료, 교재, 커리큘럼에 기반해서 설명해야 할 이론/개념/수업 내용 관련 질문
      - 'code'    : 코드 오류, 에러 메시지, 디버깅, 리팩토링, 구현 방법 등 코드 중심 질문
      - 'general' : 잡담, 자기소개 요청, 일반적인 설명, 강의자료와 직접 상관없는 일반 지식 질문

      반드시 아래 셋 중 하나만 소문자로 출력해라:
      rag
      code
      general

      질문:
      {question}
      """.strip()
    )

    # question(str) → {"question": question} → router_prompt → llm → str("rag"/"code"/"general")
    base_chain = (
        {"question": RunnablePassthrough()}
        | router_prompt
        | llm
        | StrOutputParser()
    )

    def llm_route(question: str) -> str:
        # LLM이 생성한 분류 결과 문자열을 후처리하여 rag/code/general 중 하나로 정규화
        text = base_chain.invoke(question)
        t = (text or "").strip().lower()
        if not t:
            return "general"
        first = t.split()[0]
        if first in {"rag", "code", "general"}:
            return first
        return "general"

    # 3) 최종 라우터: 규칙 → 필요하면 LLM
    def final_router(question: str) -> str:
        # 먼저 rule_based_router로 빠른 분기
        rule = rule_based_router(question)
        if rule in {"rag", "code"}:
            # code 또는 rag로 확실하게 판단되면 그대로 사용
            return rule
        # 애매한 경우에만 LLM에게 분류를 맡김
        return llm_route(question)

    # RunnableLambda로 감싸서, LCEL에서 다른 노드처럼 .invoke(question)으로 사용할 수 있게 함
    return RunnableLambda(final_router)


# 인터랙티브 질의 함수 =====
def chat_loop(rag_chain, retriever, general_chain, code_chain, router_chain):
    print("\n강의 자료 기반 + 일반 지식 + 코드 헬프를 지원하는 튜터입니다.")
    print("종료하려면 'quit' 또는 'q' 를 입력하세요.\n")

    while True:
        question = input("질문: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        if not question:
            continue

        # 0) 질의 라우팅 (에이전트 브레인)
        # router_chain: question(str) → "rag"/"code"/"general"
        route = router_chain.invoke(question)
        print(f"\n[DEBUG] 선택된 라우트: {route}")

        # 1) 라우트에 따라 적절한 체인 선택 후 실행
        if route == "rag":
            # RAG 체인을 통해 강의자료 기반 답변 생성
            answer = rag_chain.invoke(question)
            # 어떤 chunk를 참고했는지 교육용으로 함께 출력 (v1 스타일: retriever.invoke())
            docs = retriever.invoke(question)

            print("\n--- 튜터의 RAG 답변 (강의자료 기반) ---")
            print(answer)

            print("\n--- 참고한 자료(상위 몇 개 chunk) ---")
            for i, d in enumerate(docs, start=1):
                src = os.path.basename(d.metadata.get("source", "unknown"))
                print(f"[{i}] 파일: {src}")
                # chunk 내용이 너무 길 수 있으므로, 앞부분 몇 줄만 미리보기로 출력
                preview = d.page_content.strip().split("\n")
                preview = "\n".join(preview[:4])
                print(f"    {preview}")
                print()
            print("=" * 60)

        elif route == "code":
            # 코드/에러 관련 질문은 코드 헬프 체인 사용
            answer = code_chain.invoke(question)
            print("\n--- 코드 헬프/디버깅 답변 ---")
            print(answer)
            print("=" * 60)

        else:  # "general"
            # 그 외는 일반 설명/잡담 체인으로 처리
            answer = general_chain.invoke(question)
            print("\n--- 일반 설명/잡담 답변 ---")
            print(answer)
            print("=" * 60)

if __name__ == "__main__":
    # 1) 벡터 스토어 준비 (처음 한 번은 인덱스 생성, 이후부터는 디스크에서 로드)
    vectorstore = get_or_create_vectorstore()

    # 2) LLM 준비
    llm = build_llm()

    # 3) 체인들 구성
    rag_chain, retriever = build_rag_chain(vectorstore, llm)
    general_chain = build_general_chain(llm)
    code_chain = build_code_chain(llm)
    router_chain = build_question_router(llm)

    # 4) 대화 루프 시작 (에이전트 라우팅)
    chat_loop(rag_chain, retriever, general_chain, code_chain, router_chain)