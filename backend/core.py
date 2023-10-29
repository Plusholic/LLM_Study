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
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
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
    docsearch = FAISS.load_local("faiss_index_GNN", embeddings)
    # chat = ChatOpenAI(verbose=True, temperature=0)

    #
    custom_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry'.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

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
    print(run_llm(query="What is DropEdge"))