
import os
import json
import glob
import asyncio
import datetime
import tempfile
from pathlib import Path
from functools import lru_cache

import requests
import pdfplumber
import docx
import faiss
import numpy as np
from dotenv import load_dotenv
from msal import ConfidentialClientApplication

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sow_estimator import router as estimator_router


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
        raise Exception(f"Authentication failed: {result}")
    return result["access_token"]


def extract_text(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            with pdfplumber.open(tmp.name) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif ext == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            doc = docx.Document(tmp.name)
            return "\n".join(p.text for p in doc.paragraphs)

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


def embed_texts(texts, batch_size: int = 64):
    """
    Embeds in batches to reduce peak RAM and request size.
    Note: This uses the legacy openai-python interface (openai.Embedding.create).
    """
    import openai
    openai.api_key = OPENAI_API_KEY

    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
        all_vectors.extend([d["embedding"] for d in response["data"]])
    return all_vectors


def log_sync_activity(filename, user_name, user_email):
    timestamp = datetime.datetime.utcnow().isoformat()
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{filename},{user_name},{user_email}\n")


def sync_sharepoint():
    internal_subfolder = "Internal"
    public_subfolder = "Public"

    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    def list_files(sub_path):
        full_path = f"{FOLDER_PATH}/{sub_path}".replace(" ", "%20")
        url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{full_path}:/children"
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json().get("value", [])

    internal_files = list_files(internal_subfolder)
    public_files = list_files(public_subfolder)

    all_files = [(f, "internal") for f in internal_files] + [(f, "public") for f in public_files]

    for file, access_level in all_files:
        name = file.get("name")
        if not name:
            continue

        ext = name.lower().split(".")[-1]
        if ext not in ["pdf", "docx"]:
            print(f"Skipping unsupported file: {name}")
            continue

        base_name = os.path.splitext(name)[0]
        subfolder = Path(DESTINATION_FOLDER) / access_level
        subfolder.mkdir(parents=True, exist_ok=True)

        json_path = subfolder / f"{base_name}.json"
        index_path = subfolder / f"{base_name}.index"

        if json_path.exists() and index_path.exists():
            print(f"Skipping already processed file: {name}")
            continue

        print(f"Downloading: {name}")
        download_url = file.get("@microsoft.graph.downloadUrl")
        if not download_url:
            print(f"Missing download URL for: {name}")
            continue

        file_data = requests.get(download_url, timeout=120)
        file_data.raise_for_status()

        dest_path = subfolder / name
        with open(dest_path, "wb") as f:
            f.write(file_data.content)

        text = extract_text(name, file_data.content)
        chunks = chunk_text(text)

        if not chunks:
            print(f"No text extracted for: {name}")
            continue

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

    print("Sync complete. Files processed and saved to ./documents/")


# Load internal user list
INTERNAL_USER_FILE = "internal_users.json"
if os.path.exists(INTERNAL_USER_FILE):
    with open(INTERNAL_USER_FILE, "r", encoding="utf-8") as f:
        INTERNAL_USERS = json.load(f)
else:
    INTERNAL_USERS = {}


app = FastAPI()
app.include_router(estimator_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sync_in_progress = False


# =========================
# Step 3: Cached index loader
# =========================
@lru_cache(maxsize=8)
def load_folder_indexes(folder: str):
    """
    Loads all {doc}.json + {doc}.index pairs in a folder once, then caches them.
    Cache must be cleared after upload or sync.
    Returns: List[(chunks: List[str], index: faiss.Index)]
    """
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


def clear_index_cache():
    try:
        load_folder_indexes.cache_clear()
        print("Index cache cleared.")
    except Exception as e:
        print("Failed to clear index cache:", e)


# =========================
# Upload
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename
        document_name = os.path.splitext(filename)[0]

        content = await file.read()
        text = extract_text(filename, content)
        chunks = chunk_text(text)
        if not chunks:
            return {"error": "No text extracted from uploaded file."}

        vectors = embed_texts(chunks)

        subfolder = Path(DESTINATION_FOLDER) / "public"
        subfolder.mkdir(parents=True, exist_ok=True)

        with open(subfolder / f"{document_name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(np.array(vectors).astype("float32"))
        faiss.write_index(index, str(subfolder / f"{document_name}.index"))

        # New doc added => clear cache so /ask can see it
        clear_index_cache()

        return {"message": f"Document '{document_name}' uploaded and processed."}
    except Exception as e:
        print("Upload failed:", str(e))
        return {"error": str(e)}


# =========================
# Sync now (background)
# =========================
@app.post("/sync-now")
def trigger_sync(background_tasks: BackgroundTasks):
    global sync_in_progress
    if sync_in_progress:
        return {"status": "busy", "message": "Sync already in progress."}

    sync_in_progress = True

    def run_and_release():
        global sync_in_progress
        try:
            sync_sharepoint()
        finally:
            sync_in_progress = False
            clear_index_cache()

    background_tasks.add_task(run_and_release)
    return {"status": "started", "message": "SharePoint sync started in background."}


# =========================
# Ask
# =========================
@app.post("/ask")
def ask_question(question: str = Form(...), user_email: str = Form(...)):
    try:
        access_folders = ["documents/public"]

        # RULE 1 — all @erp-is.com emails can access internal docs
        if user_email and user_email.endswith("@erp-is.com"):
            access_folders.append("documents/internal")

        # RULE 2 — emails added via /admin also have internal access
        elif INTERNAL_USERS.get(user_email):
            access_folders.append("documents/internal")

        print("[DEBUG] ask_question called with:", user_email)
        print("[DEBUG] access_folders:", access_folders)

        # Embed question
        question_vec = embed_texts([question])[0]
        combined_chunks = []

        # FAISS search across allowed folders
        for folder in access_folders:
            data = load_folder_indexes(folder)
            for chunks, index in data:
                D, I = index.search(
                    np.array([question_vec]).astype("float32"),
                    k=3
                )
                for score, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(chunks):
                        combined_chunks.append((score, chunks[idx]))

        combined_chunks.sort(key=lambda x: x[0])
        top_chunks = [chunk for _, chunk in combined_chunks[:5]]

        if not top_chunks:
            return {"answer": "No relevant content found."}

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


# =========================
# Admin pages
# =========================
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
    with open(INTERNAL_USER_FILE, "w", encoding="utf-8") as f:
        json.dump(INTERNAL_USERS, f, indent=2)

    clear_index_cache()
    return HTMLResponse(f"<p>{email} added as internal user. <a href='/admin'>Back</a></p>")


@app.post("/admin/remove")
def remove_internal_user(email: str = Form(...)):
    if email in INTERNAL_USERS:
        del INTERNAL_USERS[email]
        with open(INTERNAL_USER_FILE, "w", encoding="utf-8") as f:
            json.dump(INTERNAL_USERS, f, indent=2)

        clear_index_cache()
        return HTMLResponse(f"<p>{email} removed. <a href='/admin'>Back</a></p>")
    return HTMLResponse(f"<p>{email} not found. <a href='/admin'>Back</a></p>")


# =========================
# Slack integration
# =========================
def post_to_slack(response_url: str, text: str):
    try:
        requests.post(
            response_url,
            json={"response_type": "in_channel", "text": text},
            timeout=10
        )
    except Exception as e:
        print("[DEBUG][SLACK] error posting follow-up:", e)


def process_slack_question(question: str, response_url: str):
    user_email = "default@erp-is.com"  # internal identity for Slack
    print("[DEBUG][SLACK] BG processing question:", question, "from", user_email)

    try:
        answer = ask_question(question=question, user_email=user_email)
        text = answer.get("answer") or answer.get("error") or "No answer from backend."
    except Exception as e:
        text = f"Error while processing: {str(e)}"

    if response_url:
        post_to_slack(response_url, text)


@app.post("/ask-from-slack")
async def ask_from_slack(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    question = form.get("text") or ""
    response_url = form.get("response_url")

    print("[DEBUG][SLACK] incoming text:", question)
    print("[DEBUG][SLACK] response_url:", response_url)

    if response_url:
        background_tasks.add_task(process_slack_question, question, response_url)

    return {"response_type": "ephemeral", "text": "Got it, I’m generating an answer…"}


# =========================
# Startup: sync SharePoint (kept)
# =========================
@app.on_event("startup")
async def startup_event():
    # Start sync after the app boots (background)
    asyncio.create_task(startup_sync())


async def startup_sync():
    global sync_in_progress
    if sync_in_progress:
        print("Startup sync skipped: already in progress.")
        return

    sync_in_progress = True
    try:
        print("Running SharePoint sync on startup (background)...")
        await asyncio.to_thread(sync_sharepoint)
        clear_index_cache()
        print("Startup sync finished.")
    except Exception as e:
        print(f"Startup sync failed: {e}")
    finally:
        sync_in_progress = False
