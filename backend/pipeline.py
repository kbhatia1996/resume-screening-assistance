import os
from dotenv import load_dotenv
load_dotenv()
import json
import glob
import boto3  # For S3 in prod
import chromadb
from typing import List

# Embedding provider selection order:
# 1) Hugging Face Inference (if HF_API_TOKEN set)
# 2) OpenAI (if OPENAI_API_KEY set)
# 3) Local sentence-transformers (optional, heavy)

get_embedding = None

# --- Hugging Face Inference API ---
HF_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if HF_TOKEN:
    try:
        from huggingface_hub import InferenceApi
        _hf_infer = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN)
        def get_embedding(text: str) -> List[float]:
            resp = _hf_infer(inputs=text)
            # If response is dict with 'embedding'
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
        print(f"[pipeline] Hugging Face inference init failed: {e}")
        get_embedding = None

# --- OpenAI fallback ---
if get_embedding is None and os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
        def get_embedding(text: str) -> List[float]:
            resp = _openai_client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=text
            )
            return resp.data[0].embedding
    except Exception as e:
        print(f"[pipeline] OpenAI client init failed: {e}")
        get_embedding = None

# --- Local sentence-transformers (optional heavy) ---
if get_embedding is None:
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(os.getenv("ST_MODEL", "all-MiniLM-L6-v2"))
        def get_embedding(text: str) -> List[float]:
            vec = _st_model.encode(text)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception:
        get_embedding = None
        print("[pipeline] No embedding provider available. Set HF_API_TOKEN or OPENAI_API_KEY, or install sentence-transformers (and torch) for local embeddings.")

# === Config ===
PARSED_DIR = "data/resumes/parsed/"  # local parsed resumes

# S3 configs (Prod)
S3_BUCKET_PARSED = "my-parsed-resumes"  

# Initialize S3 client (Prod only)
s3_client = boto3.client("s3")

# Initialize Chroma DB client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Create / get collection
collection = chroma_client.get_or_create_collection(name="resumes")


def process_resume(file_path, file_name):
    """Read JSON, embed, and store in Chroma."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Create embedding using summary or experience
    # If 'experience' is a list, join into text
    experience = data.get("experience", "")
    if isinstance(experience, list):
        experience = "\n".join([e.get("description", str(e)) if isinstance(e, dict) else str(e) for e in experience])
    content = (data.get("summary", "") or "") + "\n" + (experience or "")
    if get_embedding is None:
        print(f"[pipeline] Skipping embedding for {file_name}: no embedding provider configured.")
        return

    vector = get_embedding(content)

    # Store in Chroma
    collection.add(
        ids=[file_name],  # use filename as unique id
        embeddings=[vector],
        documents=[content],
        metadatas=[{"name": data.get("name"), "email": data.get("email")}]
    )

    print(f"Embedded + stored in Chroma: {file_name}")


def main():
    # --- Prod: Load from S3 ---
    # response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PARSED)
    # for obj in response.get("Contents", []):
    #     key = obj["Key"]
    #     file_name = os.path.basename(key)
    #     local_path = os.path.join(PARSED_DIR, file_name)
    #     s3_client.download_file(S3_BUCKET_PARSED, key, local_path)
    #     process_resume(local_path, file_name)

    # --- Dev: Load from local folder ---
    json_files = glob.glob(os.path.join(PARSED_DIR, "*.json"))
    for json_path in json_files:
        file_name = os.path.basename(json_path)
        process_resume(json_path, file_name)


if __name__ == "__main__":
    main()
