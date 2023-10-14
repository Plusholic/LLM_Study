from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


# >> streamlit run main.py

st.header("Chat With Graph Neural Network References")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


# 이거 한번 뜯어봐야겠는데?
# 여기에 page 추가하고싶음.
# Notion API 하는법 알고싶음
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


# 만약 프롬프트가 들어오면 다음 내용을 실행.
if prompt:
    
    # 프롬프트가 들어오면, Search the Database.. 가 나타나면서 바퀴가 돌아감
    with st.spinner("Search the Database.."):
        
        # 답변 생성
        generated_response = run_llm(query=prompt) 
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        # session_state에 질문과 답변 저장
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
