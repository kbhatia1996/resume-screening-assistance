import os
import json
import glob
import boto3   # For S3 in prod
from PyPDF2 import PdfReader

# === Config ===
RAW_DIR = "data/resumes/raw/"        # local raw resumes
PARSED_DIR = "data/resumes/parsed/"  # local parsed resumes

# S3 configs (Prod)
S3_BUCKET_RAW = "my-raw-resumes"       # bucket for raw resumes
S3_BUCKET_PARSED = "my-parsed-resumes" # bucket for parsed resumes

# Initialize S3 client (only needed in prod)
s3_client = boto3.client("s3")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF (simple version)."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_resume(text):
    """Convert raw text into structured JSON (simplified)."""
    return {
        "name": "Unknown",   # TODO: add NLP later
        "email": "Unknown",
        "phone": "Unknown",
        "skills": [],
        "experience": text[:500],  # just a snippet for now
        "education": "Unknown",
        "summary": text[:200]      # snippet for embedding
    }

def process_resume(file_path, file_name):
    """Parse one resume PDF and save JSON output."""
    text = extract_text_from_pdf(file_path)
    parsed = parse_resume(text)

    # --- Save to S3 (Prod) ---
    s3_client.put_object(
        Bucket=S3_BUCKET_PARSED,
        Key=file_name.replace(".pdf", ".json"),
        Body=json.dumps(parsed)
    )

    # --- Save locally (Dev) ---
    # out_path = os.path.join(PARSED_DIR, file_name.replace(".pdf", ".json"))
    # with open(out_path, "w") as f:
    #     json.dump(parsed, f, indent=2)

    print(f"Parsed and saved: {file_name}")

def main():
    # --- Get files from S3 (Prod) ---
    # response = s3_client.list_objects_v2(Bucket=S3_BUCKET_RAW)
    # for obj in response.get("Contents", []):
    #     key = obj["Key"]
    #     file_name = os.path.basename(key)
    #     local_path = os.path.join(RAW_DIR, file_name)
    #     s3_client.download_file(S3_BUCKET_RAW, key, local_path)
    #     process_resume(local_path, file_name)

    # --- Get files locally (Dev) ---
    pdf_files = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        process_resume(pdf_path, file_name)

if __name__ == "__main__":
    main()
