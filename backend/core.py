import os
import sys
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

sys.path.append('./')

from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local("faiss_index_react", embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="refine",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is DropEdge"))