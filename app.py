from typing import Set
import time
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
# 프롬프트 템플릿 추가. -> done

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "\n\n reference sources:\n"
    # sources_page = "page:\n"
    
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. Path : {source[0]}\n Page : {source[1]}\n"
        # sources_string += "{num} [link]({link}) \n Page : {page}\n".format(num = i+1, link=source[0], page=source[1])
        # sources_string += "[link]({link})".format(link=source[0])
        
    return sources_string
    # return source[0], source[1]


st.header("Plusholic Knowledge Database")
st.markdown("---")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    
if "db_type" not in st.session_state:
    st.session_state.db_type = "PDF papers"  # 초기 데이터베이스 유형
if "k" not in st.session_state:
    st.session_state.k = 4  # 초기 슬라이더 값 (k)
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5  # 초기 슬라이더 값 (threshold)



with st.sidebar:
    st.title('🤗💬 Plusholic Data Chat')
    
    # st.checkbox("Disable selectbox widget", key="disabled")
    db_type = st.radio(
                "Select Database Type 👉",
                key="visibility",
                options=["PDF papers", "Notion", "Excel", "PPT"],
                )
    if db_type == "PDF papers":
        # k = st.slider('How many references', 1, 20, 4)
        # threshold = st.slider('How much scores', value=0.5, min_value=0.0, max_value=1.0, step=0.05)
        st.session_state.k = st.slider('How many references', 1, 20, 4)
        st.session_state.threshold = st.slider('How much scores', value=0.5, min_value=0.0, max_value=1.0, step=0.05)


       # 채팅 내역 있으면 보이게 해야함
       
    # Database Type을 PDF Papers로 했다면,
if db_type == "PDF papers":
    source_dict = {}
    st.title("Chat With Graph Neural Network References")
    
    # k = st.slider('How many references', 1, 50, 1)
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Ask me a question about Graph Neural Network!"})
        
        
        with st.chat_message(st.session_state.messages[0]["role"]):
            # st.write(st.session_state.messages[0]["content"])
            st.markdown(st.session_state.messages[0]["content"])
            # 여기에 추가
            
    # 만약 프롬프트가 들어오면 다음 내용을 실행.
    if prompt := st.chat_input("Enter your question"):

        # 이전 메세지들 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # st.markdown(message["content"])

        # 내가 입력한 메세지가 먼저 나오도록 세팅
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(st.session_state.messages[-1]["role"]):
            # st.write(st.session_state.messages[-1]["content"])
            st.markdown(st.session_state.messages[-1]["content"])
        # 답변을 생성할때 스피너가 돌아가도록 세팅
        with st.spinner("Thinking..."):
            try:
                generated_response = run_llm(query=prompt,
                                             k=st.session_state.k,
                                             threshold=st.session_state.threshold)
            except:
                generated_response = {"source_documents" : "None",
                                      "answer" : "I dont know"}
            
        # 메세지가 챗봇으로부터 온게 아니라면 -> 유저로부터 온거라면, 응답을 생성함
        if st.session_state.messages[-1]["role"] != "assistant":
            
            # with chat_message() -> 채팅창에서 streamlit의 봇 프로필같은거 나오게함
            with st.chat_message("assistant"):
                    # 답변 생성
                    try:
                        sources = set([(doc.metadata['source'], doc.metadata['page']) for doc in generated_response["source_documents"]])
                    except:
                        sources = [("Not exists", "")]
                    # formatted_response = (f"{generated_response['result']} \n {create_sources_string(sources)}")
                    # formatted_response = (f"{generated_response['answer']} \n {create_sources_string(sources)}")

                    # formatted response 라는 말도 필요 없는거같은데? 이거 바꾸자
                    formatted_response = (f"{generated_response['answer']}")
                    
                    # st.write(formatted_response)
                    
                    full_response = ""
                    message_placeholder = st.empty()
                    
                    # 답변 라이브 스트리밍 -> OpenAI에서 streaming=True 일 때 어떻게 쓰는지 나중에 추가
                    for idx, chunk in enumerate(formatted_response.split()):  
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                        prev_chunk = chunk
                        
                    full_response += create_sources_string(sources)
                    # full_response += formatted_response
                    # source_path, source_page = create_sources_string(sources)
                    # source_dict[source_page].append(source_page)
                    
                    # 이거 검색 가능하게 바꾸자
                    from streamlit_file_browser import st_file_browser
                    # print(create_sources_string(sources)[4:-6])
                    # 이거 어떻게 수정하지..
                    event = st_file_browser(
                        key='S',
                        path='/Users/jeonjunhwi/문서/Projects/GNN_Covid/refference/GNN논문/Data-Driven',
                        show_choose_file=True,
                        show_download_file=False)
                    st.write(event)
                    
                    message_placeholder.markdown(full_response)
                    # st.markdown(create_sources_string(sources), unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response}) # Add response to message history


        # session_state에 질문과 답변 저장
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

elif db_type == "Notion":
    
    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)
    
    link='check out this [link](https://retailscope.africa/)'
    st.markdown(link,unsafe_allow_html=True)
    
elif db_type == "Excel":

    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)
    
elif db_type == "PPT":

    st.title("Chat With Graph Neural Network References")
    st.write('You Must select PDF papers')
    st.write('You current selected:', db_type)