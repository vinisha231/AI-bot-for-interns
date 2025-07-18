from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qa_chain import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_bot(query: Query):
    try:
        print(" Received question:", query.question)
        response = get_answer(query.question)
        print("Got response:", response)
        return {"answer": response}
    except Exception as e:
        print("ERROR in /ask:", str(e))
        return {"answer": "Server error: " + str(e)}
