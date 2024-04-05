from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from my_retriever import MyRetriever
def load_chain():
    # 加载问答链
    
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/DeepLearningAssistant/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/DeepLearning'

    # 加载数据库
    vector_db = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    llm = InternLM_LLM(model_path = "/root/DeepLearningAssistant/model/internlm2-chat-7b")

    # 你可以修改这里的 prompt template 来试试不同的问答效果
    template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
    提供的上下文：
    ···
    {context}
    ···
    用户的问题: {question}
    你给的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)
    
    # 加载自己的Retriever
    retriever = vector_db.as_retriever()
    myRetriever = MyRetriever(base_retriever = retriever)

    # 运行 chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=myRetriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain


qa_chain = load_chain()
while True:
    question = input("请输入你的问题：")
    if question == 'exit':
        break
    result = qa_chain(question)
    print(result)