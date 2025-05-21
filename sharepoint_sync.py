import os
import requests
from msal import ConfidentialClientApplication
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber, docx, json
import faiss
import numpy as np

# === Load secrets from environment ===
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === SharePoint Info ===
DRIVE_ID = "b!rsvKwTOMNUeCRjsRDMZh-kprgBi3tc1LiWVKbiOrmtWWapTcFH-5QLtKqb12SEmT"
FOLDER_PATH = "AI"

# === Backend processing path ===
DESTINATION_FOLDER = "documents"
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# === Constants for embedding ===
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

# === Authentication ===
SCOPES = ["https://graph.microsoft.com/.default"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

def authenticate():
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise Exception("Authentication failed.")
    return result["access_token"]

def extract_text(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(content)
        with pdfplumber.open("temp.pdf") as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == "docx":
        with open("temp.docx", "wb") as f:
            f.write(content)
        doc = docx.Document("temp.docx")
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return ""

def chunk_text(text: str, max_chars: int = 1000):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chars:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def embed_texts(texts):
    import openai
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(input=texts, model=EMBEDDING_MODEL)
    return [d["embedding"] for d in response["data"]]

def sync_sharepoint():
    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}
    encoded_path = FOLDER_PATH.replace(" ", "%20")
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{encoded_path}:/children"
    drive_resp = requests.get(url, headers=headers)
    drive_resp.raise_for_status()
    files = drive_resp.json().get("value", [])

    for file in files:
        name = file.get("name")
        if name.endswith(".pdf") or name.endswith(".docx"):
            print(f"Downloading: {name}")
            download_url = file.get("@microsoft.graph.downloadUrl")
            file_data = requests.get(download_url)
            dest_path = Path(DESTINATION_FOLDER) / name
            with open(dest_path, "wb") as f:
                f.write(file_data.content)

            # === Process file for embeddings ===
            content = file_data.content
            text = extract_text(name, content)
            chunks = chunk_text(text)
            vectors = embed_texts(chunks)

            base_name = os.path.splitext(name)[0]
            with open(Path(DESTINATION_FOLDER) / f"{base_name}.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f)

            index = faiss.IndexFlatL2(VECTOR_DIM)
            index.add(np.array(vectors).astype("float32"))
            faiss.write_index(index, str(Path(DESTINATION_FOLDER) / f"{base_name}.index"))

    print("✅ Sync complete. Files processed and saved to ./documents/")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/sync-now")
def trigger_sync():
    try:
        sync_sharepoint()
        return {"status": "success", "message": "SharePoint sync completed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

try:
    print("🚀 Running SharePoint sync on startup...")
    sync_sharepoint()
except Exception as e:
    print(f"❌ SharePoint sync failed: {e}")
