import os
import requests
from msal import ConfidentialClientApplication
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from sow_estimator import router as estimator_router
app.include_router(estimator_router)
import pdfplumber, docx, json
import faiss
import numpy as np
import datetime
import glob

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

HISTORY_LOG = os.path.join(DESTINATION_FOLDER, "sync_log.csv")

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

def log_sync_activity(filename, user_name, user_email):
    timestamp = datetime.datetime.utcnow().isoformat()
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{filename},{user_name},{user_email}\n")

def get_last_log_entry():
    if not os.path.exists(HISTORY_LOG):
        return None
    with open(HISTORY_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1].strip()
        timestamp, filename, user_name, user_email = last_line.split(",")
        return {
            "timestamp": timestamp,
            "filename": filename,
            "user_name": user_name,
            "user_email": user_email
        }

def sync_sharepoint():
    internal_subfolder = "Internal"
    public_subfolder = "Public"

    def list_files(sub_path):
        full_path = f"{FOLDER_PATH}/{sub_path}".replace(" ", "%20")
        url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{full_path}:/children"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json().get("value", []), sub_path

    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    internal_files, internal_label = list_files(internal_subfolder)
    public_files, public_label = list_files(public_subfolder)
    all_files = [(f, "internal") for f in internal_files] + [(f, "public") for f in public_files]
    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}
    encoded_path = FOLDER_PATH.replace(" ", "%20")
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{encoded_path}:/children"
    drive_resp = requests.get(url, headers=headers)
    drive_resp.raise_for_status()
    files = drive_resp.json().get("value", [])

    for file, access_level in all_files:
        name = file.get("name")
        ext = name.lower().split(".")[-1]
        if ext not in ["pdf", "docx"]:
            print(f"🚫 Skipping unsupported file: {name}")
            continue

        base_name = os.path.splitext(name)[0]
        subfolder = Path(DESTINATION_FOLDER) / access_level
        subfolder.mkdir(parents=True, exist_ok=True)
        json_path = subfolder / f"{base_name}.json"
        index_path = subfolder / f"{base_name}.index"

        if json_path.exists() and index_path.exists():
            print(f"⏭️ Skipping already processed file: {name}")
            continue

        print(f"📥 Downloading: {name}")
        download_url = file.get("@microsoft.graph.downloadUrl")
        file_data = requests.get(download_url)
        dest_path = subfolder / name
        with open(dest_path, "wb") as f:
            f.write(file_data.content)

        content = file_data.content
        text = extract_text(name, content)
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(np.array(vectors).astype("float32"))
        faiss.write_index(index, str(index_path))

        user_info = file.get("createdBy", {}).get("user", {})
        user_name = user_info.get("displayName", "Unknown")
        user_email = user_info.get("email", "Unknown")
        log_sync_activity(name, user_name, user_email)

    print("✅ Sync complete. Files processed and saved to ./documents/")

# Load internal user list
INTERNAL_USER_FILE = "internal_users.json"
if os.path.exists(INTERNAL_USER_FILE):
    with open(INTERNAL_USER_FILE, "r") as f:
        INTERNAL_USERS = json.load(f)
else:
    INTERNAL_USERS = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename
        document_name = os.path.splitext(filename)[0]

        content = await file.read()
        text = extract_text(filename, content)
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)

        subfolder = Path(DESTINATION_FOLDER) / "public"
        subfolder.mkdir(parents=True, exist_ok=True)
        with open(subfolder / f"{document_name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(np.array(vectors).astype("float32"))
        faiss.write_index(index, str(subfolder / f"{document_name}.index"))

        return {"message": f"Document '{document_name}' uploaded and processed."}
    except Exception as e:
        print("❌ Upload failed:", str(e))
        return {"error": str(e)}
# ... [rest of the code unchanged] ...

@app.post("/sync-now")
def trigger_sync():
    try:
        sync_sharepoint()
        return {"status": "success", "message": "SharePoint sync completed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
def ask_question(question: str = Form(...), user_email: str = Form(...)):
    try:
        def load_chunks_and_index(folder):
            results = []
            paths = glob.glob(f"{folder}/*.json")
            for json_path in paths:
                base = os.path.splitext(os.path.basename(json_path))[0]
                index_path = os.path.join(folder, f"{base}.index")
                if not os.path.exists(index_path):
                    continue
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                index = faiss.read_index(index_path)
                results.append((chunks, index))
            return results

        access_folders = ["documents/public"]
        if INTERNAL_USERS.get(user_email):
            access_folders.append("documents/internal")

        question_vec = embed_texts([question])[0]
        combined_chunks = []

        for folder in access_folders:
            data = load_chunks_and_index(folder)
            for chunks, index in data:
                D, I = index.search(np.array([question_vec]).astype("float32"), k=3)
                for score, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(chunks):
                        combined_chunks.append((score, chunks[idx]))

        combined_chunks.sort(key=lambda x: x[0])
        top_chunks = [chunk for _, chunk in combined_chunks[:5]]

        if not top_chunks:
            return {"answer": "No relevant content found."}
        print("Top Chunks:", top_chunks)

        prompt = f"""You are a helpful assistant answering questions using company documentation.

Based on the content provided below, answer the user's question clearly and concisely.
If the answer is spread across multiple points, synthesize the key info into a complete explanation.
Do not mention documents or chunking. Just answer as if you know the topic.

Document Content:
{chr(10).join(top_chunks)}

Question: {question}
Answer:"""

        import openai
        openai.api_key = OPENAI_API_KEY
        print(f"Processing question: {question} from {user_email}")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You answer questions based on documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    users = "<br>".join([f"{email}" for email in INTERNAL_USERS.keys()])
    return f"""
        <h2>Internal Users</h2>
        <p>{users or 'No users added yet.'}</p>
        <form method='post' action='/admin/add'>
            <input name='email' placeholder='Add user email' required>
            <button type='submit'>Add User</button>
        </form>
        <form method='post' action='/admin/remove'>
            <input name='email' placeholder='Remove user email' required>
            <button type='submit'>Remove User</button>
        </form>
    """

@app.post("/admin/add")
def add_internal_user(email: str = Form(...)):
    INTERNAL_USERS[email] = True
    with open(INTERNAL_USER_FILE, "w") as f:
        json.dump(INTERNAL_USERS, f, indent=2)
    return HTMLResponse(f"<p>{email} added as internal user. <a href='/admin'>Back</a></p>")

@app.post("/admin/remove")
def remove_internal_user(email: str = Form(...)):
    if email in INTERNAL_USERS:
        del INTERNAL_USERS[email]
        with open(INTERNAL_USER_FILE, "w") as f:
            json.dump(INTERNAL_USERS, f, indent=2)
        return HTMLResponse(f"<p>{email} removed. <a href='/admin'>Back</a></p>")
    return HTMLResponse(f"<p>{email} not found. <a href='/admin'>Back</a></p>")

@app.get("/sync-latest")
def get_sync_latest():
    latest = get_last_log_entry()
    if not latest:
        return JSONResponse(status_code=404, content={"error": "No sync history found."})
    return latest

@app.post("/ask-from-slack")
async def ask_from_slack(request: Request):
    form = await request.form()
    question = form.get("text")
    user_email = form.get("user_email") or "default@shiperp.com"  # fallback
    try:
        answer = ask_question(question=question, user_email=user_email)
        return {"response_type": "in_channel", "text": answer["answer"]}
    except Exception as e:
        return {"response_type": "ephemeral", "text": f"Error: {str(e)}"}


try:
    print("🚀 Running SharePoint sync on startup...")
    sync_sharepoint()
except Exception as e:
    print(f"❌ SharePoint sync failed: {e}")
