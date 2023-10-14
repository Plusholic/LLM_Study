# 환경변수 준비

# pip install torch transformers python-pptx Pillow
# pip install pypdf
import os

from llama_index import SimpleDirectoryReader
import logging
import sys
from llama_index import GPTVectorStoreIndex
from llama_index import StorageContext, load_index_from_storage

# 로그 레벨 설정
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
# 문서 로드(data 폴더에 문서를 넣어 두세요)
# documents = SimpleDirectoryReader("/Users/jeonjunhwi/문서/Projects/GNN_Covid/refference/GNN논문").load_data()

# 인덱스 생성
# index = GPTVectorStoreIndex.from_documents(documents)


# 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
# 쿼리 엔진 생성
query_engine = index.as_query_engine(
)

# 질의응답
response = query_engine.query("Please to explain Graph Attention Network(GAT)")
# streaming_response.print_response_stream()
print(f"response:\n{response}")
print(f"source_nodes:\n{response.source_nodes}")
# # 인덱스 저장
# index.storage_context.persist()

# docid로 원래 데이터 찾는법...


