from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import time
from dotenv import load_dotenv
load_dotenv()

class OptimizeWebRAG:
  def __init__(self):
    self.search = TavilySearch(
        max_results=5, api_key=os.getenv("TAVILY_API_KEY")
        )

    self.llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # 검색 결과 요약용 프롬프트
    self.summary_prompt = ChatPromptTemplate.from_template(
        """
          당신은 검색 결과를 정리하는 요약 전문가 입니다.
          다음은 웹 검색 결과입니다.
          {searh_results}
          요구사항 :
          - 광고/홍보성/중복 내용을 제거하고
          - 서로 겹치는 내용은 합쳐서 핵심 정보만 5줄 내외로 간결하게 정리하세요

          검색결과 요약 :
        """
    )

    # 답변용 프롬프트
    self.answer_prompt = ChatPromptTemplate.from_template(
        """
          아래는 어떤 질문에 대한 웹 검색 결과을 수행한 뒤 정리한 요약 내용입니다.
          [검색 요약] {summary}
          [질문] {question}
          위 요약에 있는 정보만 사용해 질문에 답하시오.
          추측하거나 지어내지 말고, 모르는 내용은 모른다고 하세요.
          최종 답변 :
        """
    )

  # 검색결과 요약관련 메서드
  def sumarize_search(self, question:str) -> str:
    # Tavily 검색
    raw_results = self.search.invoke({"query": question})
    time.sleep(0.2)    # 너무 빠른 연속 호출은 좋지 않다.

    chain = self.summary_prompt | self.llm
    summary_msg = chain.invoke(
        {"searh_results": raw_results}
    )
    return summary_msg.content

  # 요약 정보로 최종 답변
  def answer_question(self, question:str) -> str:
    print(f"검색 및 요약 중 : {question}")
    summary = self.sumarize_search(question)
    chain = self.answer_prompt | self.llm
    answer_msg = chain.invoke(
        {"summary": summary, "question": question}
    )
    return answer_msg.content

if __name__ == "__main__":
  rag_obj = OptimizeWebRAG()

  questions = [
      "최신 AI 기술 동향은?",
      "한국에서 가장 인기 있는 빵은?",
      "한국의 방산 관련 최근 이슈는?"
  ]

  for q in questions:
    print("\n-------------------")
    print(f"질문 : {q}")
    answer = rag_obj.answer_question(q)
    print(f"최종 답변 : {answer}\n")