from fastapi import FastAPI, Query
from pydantic import BaseModel
import chromadb
from openai import OpenAI

# ---------- Config ----------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "resumes"
OPENAI_MODEL = "text-embedding-3-small"  # change to larger if needed

# ---------- Init ----------
app = FastAPI()
client = OpenAI()
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

# ---------- Request/Response Models ----------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    summarize: bool = False

class QueryResponse(BaseModel):
    query: str
    results: list
    summary: str | None = None

# ---------- API ----------
@app.post("/search", response_model=QueryResponse)
def search_resumes(req: QueryRequest):
    # 1. Embed the query
    embedding = client.embeddings.create(
        model=OPENAI_MODEL,
        input=req.query
    ).data[0].embedding

    # 2. Retrieve from Chroma
    results = collection.query(
        query_embeddings=[embedding],
        n_results=req.top_k
    )

    # Format results
    docs = [
        {
            "id": results["ids"][0][i],
            "resume_snippet": results["documents"][0][i],
            "score": results["distances"][0][i]
        }
        for i in range(len(results["ids"][0]))
    ]

    # 3. Optional: Summarize results with OpenAI
    summary = None
    if req.summarize and docs:
        resume_texts = "\n\n".join([d["resume_snippet"] for d in docs])
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a recruiter assistant."},
                {"role": "user", "content": f"Summarize these resumes for query '{req.query}':\n{resume_texts}"}
            ]
        )
        summary = completion.choices[0].message.content

    return QueryResponse(
        query=req.query,
        results=docs,
        summary=summary
    )

# ---------- Run Locally ----------
# PROD: uvicorn query_api:app --host 0.0.0.0 --port 8000
# DEV: uvicorn python.query_api:app --reload --port 8000

