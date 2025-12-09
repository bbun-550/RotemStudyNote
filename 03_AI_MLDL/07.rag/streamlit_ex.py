
# !pip install streamlit sentence-transformers chromadb google-generativeai python-dotenv
# pip install --upgrade torch watchdog tf-keras
# conda activate venv 에서 실습
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# 지식 자료
knowledge = [
    "사자는 갈귀털이 매우 길다",
    "기린은 목이 길다",
    "치타는 지구상에서 가장 빠르다",
    "하마는 물속에서 생활하는 포유류다",
    "펭귄은 날지 못하나 수영은 잘한다"
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(knowledge)
print(embeddings)

# VectorDB ...
chroma_client = chromadb.Client(Settings(persist_directory="./rag_demo", anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection("animals")

for i, (text, emb) in enumerate(zip(knowledge, embeddings)):
    collection.add(
        documents=[text],
        embeddings=[emb.tolist()],
        ids=[f"doc_{i}"]
    )

all_data = collection.get()
print(all_data)

print(collection.count())
doc = collection.get(ids=["doc_0"])
print(doc)

# 질문 처리 ---------------------------------------------------------------------------------
# query = "목이 긴 동물은?"
# query_vec = embedder.encode([query])[0]
# # print(query_vec)

# Streamlit UI에서 질문 받기
import streamlit as st
st.title("LLM RAG 연습")
query = st.text_input("질문을 입력하세요")
print(f"query : {query}")
query_vec = embedder.encode([query])[0]
# streamlit 추가된 부분 ----------------------------------------------------------------------

results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=3,
    include=["documents"]
)
# print(results)

context = "\n".join(results["documents"][0])
print(f"context :\n{context}")

# LLM에게 프롬프트(검색 + 증강)에 대한 답변을 요구(생성)
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

if query:
    query_vec = embedder.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=3,
        include=["documents"]
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
    아래 정보를 참고해서 친절한 답을 해줘.
    가능하면 예시나 관련 배경지식도 함께 알려줘.
    정보 : 
    {context}
    질문 : 
    {query}
    추가사항 : 마크업은 반드시 빼줘.
    """

    with st.spinner("LLM이 답변 생성 중"):
        response = model.generate_content(prompt)
    st.subheader("LLM이 답변 결과 :")
    st.write(response.text)