import os
import json
import glob
import boto3  # For S3 in prod
import chromadb
from langchain_openai import OpenAIEmbeddings

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

# OpenAI Embeddings
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")  
# NOTE: requires OPENAI_API_KEY in env


def process_resume(file_path, file_name):
    """Read JSON, embed, and store in Chroma."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Create embedding using summary or experience
    content = data.get("summary", "") + " " + data.get("experience", "")
    vector = embedding_fn.embed_query(content)

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
