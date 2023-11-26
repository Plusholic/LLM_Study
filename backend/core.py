import os
import sys
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

sys.path.append('./')
from langchain.memory import ConversationBufferMemory
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, HypotheticalDocumentEmbedder, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# def run_llm(query: str) -> Any:
#     embeddings = OpenAIEmbeddings()
#     docsearch = FAISS.load_local("faiss_index_react", embeddings)
#     chat = ChatOpenAI(verbose=True, temperature=0)
#     qa = RetrievalQA.from_chain_type(
#         llm=chat,
#         chain_type="refine",
#         retriever=docsearch.as_retriever(),
#         return_source_documents=True,
#     )
#     return qa({"query": query})
    
def run_llm(query: str,
            k : int,
            threshold : float) -> Any:
    
    embeddings = OpenAIEmbeddings()
    # docsearch = FAISS.load_local("faiss_index_256", embeddings)
    # docsearch = FAISS.load_local("faiss_openai_512_sentence_splitter", embeddings)
    docsearch = FAISS.load_local("faiss_openai_128_sentence_splitter", embeddings)
    # chat = ChatOpenAI(verbose=True, temperature=0)

    # 답을 찾을 수 없는 경우, 답을 찾을 수 없다고 말하고 자체 지식에 의존하세요 라는 내용 추가
    custom_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry'.
    If you can't find an answer, say you can't find an answer and rely on your own knowledge.
    Chat History:
    {chat_history}
    
    Question:
    {question}
    Answers:"""

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    embeddings = OpenAIEmbeddings()
    memory = ConversationBufferMemory(memory_key="chat_history",
                                    output_key = 'answer',
                                    return_messages=True)
    
    llm = llm=ChatOpenAI(verbose=True,
                         temperature=0,
                        #  streaming=True,
                        #  callbacks=[StreamingStdOutCallbackHandler()]
                        )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k":k,
                "score_threshold": threshold
                }
            ),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        return_source_documents=True,
        chain_type='refine',
        memory=memory
    )
    
    # chat_history = []
    return qa({"question": query})



if __name__ == "__main__":
    print(run_llm(query="What is DropEdge", k=3, threshold=0.5))