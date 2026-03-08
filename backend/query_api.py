import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Query
from pydantic import BaseModel
import chromadb
from typing import List

# Embedding provider selection order:
# 1) Hugging Face Inference (if HF_API_TOKEN set)
# 2) OpenAI (if OPENAI_API_KEY set)
# 3) Local sentence-transformers (optional, heavy)

get_embedding = None

HF_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if HF_TOKEN:
    try:
        from huggingface_hub import InferenceApi
        _hf_infer = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN)
        def get_embedding(text: str):
            resp = _hf_infer(inputs=text)
            if isinstance(resp, dict) and 'embedding' in resp:
                return resp['embedding']
            if isinstance(resp, (list, tuple)):
                return list(resp)
            if isinstance(resp, dict):
                for v in resp.values():
                    if isinstance(v, (list, tuple)):
                        return list(v)
            raise RuntimeError('Unexpected HF Inference response format for embeddings')
    except Exception as e:
        print(f"[query_api] Hugging Face init failed: {e}")
        get_embedding = None

if get_embedding is None and os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
        def get_embedding(text: str):
            resp = _openai_client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=text
            )
            return resp.data[0].embedding
    except Exception as e:
        print(f"[query_api] OpenAI init failed: {e}")
        get_embedding = None

if get_embedding is None:
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(os.getenv("ST_MODEL", "all-MiniLM-L6-v2"))
        def get_embedding(text: str):
            vec = _st_model.encode(text)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception:
        get_embedding = None
        print("[query_api] No embedding provider available. Set HF_API_TOKEN or OPENAI_API_KEY, or install sentence-transformers (and torch) for local embeddings.")

# ---------- Config ----------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "resumes"
OPENAI_MODEL = "text-embedding-3-small"  # change to larger if needed

# ---------- Init ----------
app = FastAPI()
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
    # 1. Embed the query (OpenAI or local fallback)
    if 'get_embedding' not in globals() or get_embedding is None:
        # No embedding provider configured
        return QueryResponse(query=req.query, results=[], summary=f"No embedding provider configured. Set OPENAI_API_KEY or install sentence-transformers locally.")

    embedding = get_embedding(req.query)

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

    # 3. Optional: Summarize results with OpenAI (only when OpenAI client initialized)
    summary = None
    if req.summarize and docs and ('_openai_client' in globals()):
        resume_texts = "\n\n".join([d["resume_snippet"] for d in docs])
        completion = _openai_client.chat.completions.create(
            model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
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

