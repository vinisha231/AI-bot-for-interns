# backend/qa_chain.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# print("OPENAI_API_KEY:", openai_api_key)
# Load vectorstore (must be pre-generated via ingest_docs.py)
VECTORSTORE_DIR = "./chroma_db"

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)  # or OpenAIEmbeddings(openai_api_key="sk-...")
vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding)

# Use GPT-4o with a Retrieval QA chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

def get_answer(question: str) -> str:
    return qa_chain.run(question)
