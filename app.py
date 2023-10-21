from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


# >> streamlit run app.py
# 여기에 page 추가하고싶음. -> done
# sidebar로 Notion, Papers 등 선택할 수 있게 만들기 -> done
# Notion API 로 대답하는 챗봇 만들기 -> done
# Excel Base로 대답하는 챗봇 만들기 
# PPT Base로 대답하는 챗봇 만들기 
# 대화 내용을 기억하게 하려면, run_llm 에서 query에 뭔가를 추가해줘야 하나..?
# 프롬프트 템플릿 추가.

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "reference sources:\n"
    # sources_page = "page:\n"
    
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. Path : {source[0]}\n Page : {source[1]}\n"
        
    return sources_string


st.header("Plusholic Knowledge Database")
st.markdown("---")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    
# if "messages" not in st.session_state.keys(): # Initialize the chat message history
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about Graph Neural Network!"}
#     ]
#     with st.chat_message(st.session_state.messages[0]["role"]):
#         st.write(st.session_state.messages[0]["content"])


with st.sidebar:
    st.title('🤗💬 Plusholic Data Chat')
    
    # st.checkbox("Disable selectbox widget", key="disabled")
    db_type = st.radio(
                "Select Database Type 👉",
                key="visibility",
                options=["PDF papers", "Notion", "Excel", "PPT"],
                )
   
    # 자격증명 창 추가
    # if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
    #     st.success('HuggingFace Login credentials already provided!', icon='✅')
    #     hf_email = st.secrets['EMAIL']
    #     hf_pass = st.secrets['PASS']
    # else:
    #     hf_email = st.text_input('Enter E-mail:', type='password')
    #     hf_pass = st.text_input('Enter password:', type='password')
    #     if not (hf_email and hf_pass):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Database Type을 PDF Papers로 했다면,
if db_type == "PDF papers":
        
    st.title("Chat With Graph Neural Network References")

    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Graph Neural Network!"}
        ]
        with st.chat_message(st.session_state.messages[0]["role"]):
            # st.write(st.session_state.messages[0]["content"])
            st.markdown(st.session_state.messages[0]["content"])
            # 여기에 추가
            st.session_state.messages.append({"role": "user", "content": prompt})

    # 만약 프롬프트가 들어오면 다음 내용을 실행.
    if prompt := st.chat_input("Enter your question"):
        
        # 프롬프트가 들어오면, Search the Database.. 가 나타나면서 바퀴가 돌아감
        # with st.spinner("Search the Database.."):
        # with st.spinner("Thinking..."):
        # 답변 생성
        generated_response = run_llm(query=prompt) 
        st.session_state.messages.append({"role": "user", "content": prompt})

         # Display the prior chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # 메세지가 챗봇으로부터 온게 아니라면 -> 유저로부터 온거라면, 응답을 생성함
        if st.session_state.messages[-1]["role"] != "assistant":
            
            # with chat_message() -> 채팅창에서 streamlit의 봇 프로필같은거 나오게함
            # with :
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # 왜 spinner가 안돌아가지...
                    # 답변 생성
                    sources = set([(doc.metadata['source'], doc.metadata['page']) for doc in generated_response["source_documents"]])
                    formatted_response = (f"{generated_response['result']} \n {create_sources_string(sources)}")
                    
                    # st.write(formatted_response)
                    
                    full_response = ""
                    message_placeholder = st.empty()
                    import time
                    for chunk in formatted_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                        
                    message_placeholder.markdown(full_response)
                    # st.session_state.messages.append({"role": "assistant", "content": formatted_response}) # Add response to message history
                    st.session_state.messages.append({"role": "assistant", "content": full_response}) # Add response to message history


        # session_state에 질문과 답변 저장
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

elif db_type == "Notion":
    
    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)

elif db_type == "Excel":

    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)
    
elif db_type == "PPT":

    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)


## Session History
# if st.session_state["chat_answers_history"]:
#     for generated_response, user_query in zip(
#         st.session_state["chat_answers_history"],
#         st.session_state["user_prompt_history"],
#     ):
        # message(generated_response)
        # message(user_query, is_user=True)

# for message in st.session_state.messages: # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])