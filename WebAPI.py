from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from my_retriever import MyRetriever
from Interface import load_chain
from fastapi.middleware.cors import CORSMiddleware
llm = load_chain()
app = FastAPI()
class Question(BaseModel):
    question: str
# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/gpt/")
async def LLM(Q: Question):
    answer = llm(Q.question)
    print(answer["source_documents"])
    print(answer["result"])
    return {"code":1,"answer": answer}


if __name__ == '__main__':
    print("启动程序！")
    uvicorn.run(app,host="127.0.0.1", port=8010) #main：py文件名 app：api对象名