from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from my_retriever import MyRetriever
persist_directory = 'data_base/vector_db/DeepLearning'
embeddings = HuggingFaceEmbeddings(model_name="/root/DeepLearningAssistant/model/sentence-transformer")
# 加载数据库
vector_db = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embeddings
)
retriever = vector_db.as_retriever()
# myRetriever = MyRetriever(base_retriever = retriever)
while True:
    tmp = input("请输入你想搜索的：")
    if tmp == 'exit':
        break
    docs = retriever.get_relevant_documents(tmp)
    # print(docs)
    print(docs)