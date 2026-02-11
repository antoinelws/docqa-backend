import os, json, glob, asyncio, datetime, tempfile
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple

import bcrypt
import requests
import pdfplumber
import docx
import faiss
import numpy as np
from dotenv import load_dotenv
from msal import ConfidentialClientApplication

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_303_SEE_OTHER

from sow_estimator import router as estimator_router



# =========================
# Config
# =========================
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models (Option B constants kept, routing can be re-added once stable)
MODEL_MINI = os.getenv("OPENAI_MODEL_MINI", "gpt-5-mini")
MODEL_BIG = os.getenv("OPENAI_MODEL_BIG", "gpt-5.2")

DRIVE_ID = os.getenv(
    "SHAREPOINT_DRIVE_ID",
    "b!rsvKwTOMNUeCRjsRDMZh-kprgBi3tc1LiWVKbiOrmtWWapTcFH-5QLtKqb12SEmT",
)
FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH", "AI")

DESTINATION_FOLDER = "documents"
os.makedirs(DESTINATION_FOLDER, exist_ok=True)
HISTORY_LOG = os.path.join(DESTINATION_FOLDER, "sync_log.csv")

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

SCOPES = ["https://graph.microsoft.com/.default"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

PUBLIC_USER_FILE = "public_users.json"
INTERNAL_USER_FILE = "internal_users.json"

def _load_users(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}

INTERNAL_USERS = _load_users(INTERNAL_USER_FILE)  # email -> true OR {"password_hash": "..."}
PUBLIC_USERS = _load_users(PUBLIC_USER_FILE)      # email -> true OR {"password_hash": "..."}


FALLBACK_MARKER = "the documentation does not mention this, but here is what i know from general knowledge:"


# =========================
# OpenAI helpers
# =========================
def is_allowed(users: Dict[str, Any], email: str) -> bool:
    v = users.get(email)
    if v is True:
        return True
    if isinstance(v, dict) and isinstance(v.get("password_hash"), str) and v["password_hash"].strip():
        return True
    return False

def get_user_tier(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    if not email:
        return None
    if is_allowed(INTERNAL_USERS, email):
        return "internal"
    if is_allowed(PUBLIC_USERS, email):
        return "public"
    return None

def get_password_hash(users: Dict[str, Any], email: str) -> Optional[str]:
    v = users.get(email)
    if isinstance(v, dict):
        ph = v.get("password_hash")
        if isinstance(ph, str) and ph.strip():
            return ph.strip()
    return None

def verify_password(email: str, password: str) -> bool:
    email = (email or "").strip().lower()
    password = password or ""

    rec_hash = get_password_hash(INTERNAL_USERS, email) or get_password_hash(PUBLIC_USERS, email)
    if not rec_hash:
        return False

    return bcrypt.checkpw(password.encode("utf-8"), rec_hash.encode("utf-8"))

def chat_completion(model: str, messages: List[dict], max_completion_tokens: int = 700) -> str:
    """
    Minimal wrapper to avoid params that some endpoints reject.
    IMPORTANT: Uses legacy openai SDK style (openai.ChatCompletion.create).
    """
    import openai

    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# Auth / SharePoint
# =========================
def authenticate() -> str:
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise Exception(f"Authentication failed: {result}")
    return result["access_token"]


# =========================
# Text extraction + chunking
# =========================
def extract_text(filename: str, content: bytes) -> str:
    """Extracts text from pdf/docx/txt."""
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            with pdfplumber.open(tmp.name) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)

    if ext == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            d = docx.Document(tmp.name)
            return "\n".join(p.text for p in d.paragraphs)

    if ext == "txt":
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return ""


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Simple chunker for docs."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = text.split(". ")
    chunks, buf = [], ""
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        candidate = (buf + s + ". ").strip()
        if len(candidate) <= max_chars:
            buf = candidate + " "
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = (s + ". ").strip()
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# =========================
# Embeddings (safe)
# =========================
def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Safe embedding; skips failing batches."""
    import openai

    openai.api_key = OPENAI_API_KEY

    clean_texts: List[str] = []
    for t in texts:
        if not t or not isinstance(t, str):
            continue
        t = t.strip()
        if len(t) < 10:
            continue
        if len(t) > 6000:
            t = t[:6000]
        clean_texts.append(t)

    if not clean_texts:
        return []

    vectors: List[List[float]] = []
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i : i + batch_size]
        try:
            resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            vectors.extend([d["embedding"] for d in resp["data"]])
        except Exception as e:
            print("[EMBEDDING ERROR] batch skipped:", str(e))
    return vectors


# =========================
# Index cache
# =========================
@lru_cache(maxsize=8)
def load_folder_indexes(folder: str) -> List[Tuple[List[str], faiss.Index]]:
    """Loads all {doc}.json + {doc}.index pairs in a folder once, then caches them."""
    results: List[Tuple[List[str], faiss.Index]] = []
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


@lru_cache(maxsize=8)
def load_folder_indexes_with_names(folder: str) -> List[Tuple[str, List[str], faiss.Index]]:
    """Same as load_folder_indexes, but includes doc name (basename)."""
    results: List[Tuple[str, List[str], faiss.Index]] = []
    paths = glob.glob(f"{folder}/*.json")

    for json_path in paths:
        base = os.path.splitext(os.path.basename(json_path))[0]
        index_path = os.path.join(folder, f"{base}.index")
        if not os.path.exists(index_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        index = faiss.read_index(index_path)
        results.append((base, chunks, index))

    return results


def clear_index_cache():
    try:
        load_folder_indexes.cache_clear()
        load_folder_indexes_with_names.cache_clear()
        print("Index cache cleared.")
    except Exception as e:
        print("Failed to clear index cache:", e)


# =========================
# SharePoint sync
# =========================
def log_sync_activity(filename: str, user_name: str, user_email: str):
    ts = datetime.datetime.utcnow().isoformat()
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts},{filename},{user_name},{user_email}\n")


def sync_sharepoint():
    internal_subfolder = "Internal"
    public_subfolder = "Public"

    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    def list_files(sub_path: str):
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
        if ext not in ["pdf", "docx", "txt"]:
            print(f"Skipping unsupported file: {name}")
            continue

        base_name = os.path.splitext(name)[0]
        subfolder = Path(DESTINATION_FOLDER) / access_level
        subfolder.mkdir(parents=True, exist_ok=True)

        json_path = subfolder / f"{base_name}.json"
        index_path = subfolder / f"{base_name}.index"

        # Skip already processed
        if json_path.exists() and index_path.exists():
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
        if not vectors:
            print(f"[WARN] No embeddings generated for {name}, skipping file")
            continue

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




# =========================
# FastAPI
# =========================
app = FastAPI()
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
if not SESSION_SECRET:
    raise RuntimeError("Missing SESSION_SECRET env var")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=True,  # Render = HTTPS => True recommandé
)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    email = (request.session.get("user_email") or "").strip().lower()
    if get_user_tier(email):
        return RedirectResponse(url="/chat-ui", status_code=HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Login - ShipERP AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 520px; }
    input, button { font-size: 14px; padding: 10px; width: 100%; margin-top: 10px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; }
    .muted { color: #666; font-size: 12px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>ShipERP AI Assistant</h2>
    <form method="post" action="/login">
      <input name="email" placeholder="Email" autocomplete="username" required />
      <input name="password" placeholder="Password" type="password" autocomplete="current-password" required />
      <button type="submit">Sign in</button>
    </form>
    <div class="muted">Access is restricted to authorized users.</div>
  </div>
</body>
</html>
"""


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email_lc = (email or "").strip().lower()

    # deny-by-default
    if not get_user_tier(email_lc):
        return HTMLResponse("<p>Access denied.</p><p><a href='/login'>Back</a></p>", status_code=403)

    # vrai login: nécessite password_hash (un user en "true" ne peut pas se logguer tant que pas migré)
    if not verify_password(email_lc, password or ""):
        return HTMLResponse("<p>Invalid credentials.</p><p><a href='/login'>Back</a></p>", status_code=401)

    request.session["user_email"] = email_lc
    return RedirectResponse(url="/chat-ui", status_code=HTTP_303_SEE_OTHER)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)


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
# Upload (public)
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
        if not vectors:
            return {"error": "No valid text to embed in this document."}

        subfolder = Path(DESTINATION_FOLDER) / "public"
        subfolder.mkdir(parents=True, exist_ok=True)

        with open(subfolder / f"{document_name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(np.array(vectors).astype("float32"))
        faiss.write_index(index, str(subfolder / f"{document_name}.index"))

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
# Retrieve (with sources)
# =========================
def retrieve_chunks_with_sources(
    question: str,
    access_folders: List[str],
    k_per_doc: int = 6,
    max_total: int = 12,
    chunk_max_chars: int = 1200,
    debug: bool = False,
) -> Tuple[List[str], List[str]]:
    """Return best chunks + unique source doc basenames."""
    question = (question or "").strip()
    if not question:
        return [], []

    qvecs = embed_texts([question])
    if not qvecs:
        return [], []
    qvec = qvecs[0]

    scored: List[Tuple[float, str, str]] = []  # (dist, chunk, doc_base)

    for folder in access_folders:
        json_paths = glob.glob(f"{folder}/*.json")
        for json_path in json_paths:
            base = os.path.splitext(os.path.basename(json_path))[0]
            index_path = os.path.join(folder, f"{base}.index")
            if not os.path.exists(index_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks_list = json.load(f)
                index = faiss.read_index(index_path)
            except Exception as e:
                if debug:
                    print("[RAG][DEBUG] failed loading:", json_path, "err:", e)
                continue

            D, I = index.search(np.array([qvec]).astype("float32"), k=k_per_doc)
            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(chunks_list):
                    c = (chunks_list[idx] or "").strip()
                    if c:
                        scored.append((float(dist), c, base))

    if not scored:
        return [], []

    scored.sort(key=lambda x: x[0])

    seen_chunks = set()
    chunks: List[str] = []
    sources_ordered: List[str] = []

    for dist, c, doc_base in scored:
        if c in seen_chunks:
            continue
        seen_chunks.add(c)

        if len(c) > chunk_max_chars:
            c = c[:chunk_max_chars].rstrip() + "…"

        chunks.append(c)
        sources_ordered.append(doc_base)

        if len(chunks) >= max_total:
            break

    uniq_sources: List[str] = []
    seen_src = set()
    for s in sources_ordered:
        if s and s not in seen_src:
            seen_src.add(s)
            uniq_sources.append(s)

    if debug:
        print("[RAG][DEBUG] question:", question)
        print("[RAG][DEBUG] access_folders:", access_folders)
        print("[RAG][DEBUG] chunks:", len(chunks), "sources:", uniq_sources)

    return chunks, uniq_sources


# =========================
# One-shot QA (/ask) - docs-first + fallback (stable)
# =========================
@app.post("/ask")
def ask_question(
    question: str = Form(...),
    user_email: str = Form(...),
    debug: bool = Form(False),
):
    try:
        question = (question or "").strip()
        user_email = (user_email or "").strip()

        # Access control
        tier = get_user_tier(user_email)
        if tier == "internal":
            access_folders = ["documents/public", "documents/internal"]
        elif tier == "public":
            access_folders = ["documents/public"]
        else:
            # deny-by-default
            return JSONResponse({"error": "Access denied"}, status_code=403)



        chunks, sources = retrieve_chunks_with_sources(
            question=question,
            access_folders=access_folders,
            k_per_doc=8,
            max_total=12,
            chunk_max_chars=1200,
            debug=debug,
        )

        has_docs = bool(chunks)
        docs_block = "\n---\n".join(chunks) if chunks else "(none found for this question)"

        system_rules = (
            "You are the ShipERP assistant.\n"
            "You must answer FIRST using the documentation excerpts provided below.\n"
            "If the documentation contains relevant information, base your answer strictly on it.\n\n"
            "If the documentation does NOT contain the answer, say explicitly:\n"
            '"The documentation does not mention this, but here is what I know from general knowledge:"\n\n'
            "Only then, provide a concise general-knowledge answer.\n"
            "Do not mix documentation-based information and general knowledge in the same sentence.\n"
            "Be practical and sufficiently detailed to be useful.\n"
            "Do NOT include any 'Sources' section in your answer."
        )

        messages = [
            {"role": "system", "content": system_rules},
            {"role": "system", "content": f"Documentation excerpts:\n{docs_block}"},
            {"role": "user", "content": question},
        ]

        # Stable: big model for now
        answer = chat_completion(model=MODEL_BIG, messages=messages, max_completion_tokens=700)

        # Sources logic: only attach sources if answer is docs-based (no fallback)
        if answer.lower().startswith(FALLBACK_MARKER):
            out_sources: List[str] = []
            final_answer = answer
        else:
            out_sources = sources if has_docs else []
            if out_sources:
                final_answer = answer.rstrip() + "\n\nSources: " + ", ".join(out_sources)
            else:
                final_answer = answer

        payload: Dict[str, Any] = {"answer": final_answer, "sources": out_sources}
        if debug:
            payload["has_docs"] = has_docs
            payload["model"] = MODEL_BIG

        return payload

    except Exception as e:
        return {"error": str(e)}


# =========================
# Conversation (per-user) + RAG
# =========================
CHAT_MODEL = MODEL_BIG
CHAT_MAX_TURNS = 12
CHAT_SUMMARY_TRIGGER = 18
CHAT_CONTEXT_CHAR_BUDGET = 14000

# In-memory store (resets on restart)
CHAT_STORE: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _trim_messages_to_budget(messages: List[dict], budget_chars: int) -> List[dict]:
    out = []
    total = 0
    for m in reversed(messages):
        c = (m.get("content") or "").strip()
        if not c:
            continue
        size = len(c) + 50
        if total + size > budget_chars and out:
            break
        out.append({"role": m["role"], "content": c})
        total += size
    return list(reversed(out))


def _get_user_state(user_id: str) -> Dict[str, Any]:
    if user_id not in CHAT_STORE:
        CHAT_STORE[user_id] = {"summary": "", "messages": []}
    return CHAT_STORE[user_id]


def _summarize_if_needed(user_id: str):
    state = _get_user_state(user_id)
    msgs = state["messages"]

    if len(msgs) <= CHAT_SUMMARY_TRIGGER:
        return

    keep = msgs[-CHAT_MAX_TURNS:]
    old = msgs[:-CHAT_MAX_TURNS]
    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old if m.get("content")])

    if not old_text.strip():
        state["messages"] = keep
        return

    summary_prompt = f"""You are maintaining a running summary of a chat.

Current summary (may be empty):
{state['summary']}

New dialogue to summarize:
{old_text}

Update the summary. Keep it factual and concise. Preserve decisions, constraints, names, and open questions.
"""

    try:
        new_summary = chat_completion(
            model=MODEL_MINI,
            messages=[
                {"role": "system", "content": "You summarize chat history into a concise running memory."},
                {"role": "user", "content": summary_prompt},
            ],
            max_completion_tokens=450,
        )
        state["summary"] = new_summary.strip()
    except Exception as e:
        print("[CHAT] summarization failed:", e)

    state["messages"] = keep


@app.post("/chat-api")
async def chat_api(
    request: Request,
    message: str = Form(...),
    debug: bool = Form(False),
):
    email = (request.session.get("user_email") or "").strip().lower()
    tier = get_user_tier(email)
    if not tier:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = email

    if tier == "internal":
        access_folders = ["documents/public", "documents/internal"]
    elif tier == "public":
        access_folders = ["documents/public"]
    else:
        return JSONResponse({"error": "Access denied"}, status_code=403)

    try:
        import re

        message = (message or "").strip()
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        state = _get_user_state(user_id)
        state["messages"].append({"role": "user", "content": message, "ts": _now_iso()})
        _summarize_if_needed(user_id)

        msg_lc = message.lower().strip()

        def is_chat_memory_question(s: str) -> bool:
            patterns = [
                r"\bfirst question\b",
                r"\bmy first question\b",
                r"\bwhat was i asking\b",
                r"\bwhat did i ask\b",
                r"\bearlier\b.*\b(chat|conversation)\b",
                r"\bin this chat\b",
                r"\bin this conversation\b",
                r"\bwhat did you say\b",
                r"\bwhat did i say\b",
                r"\bwhat have we discussed\b",
                r"\brappelle\b.*\b(premi|première)\b.*\bquestion\b",
                r"\bc'était quoi\b.*\bma\b.*\b(premi|première)\b.*\bquestion\b",
                r"\bdans ce chat\b",
                r"\bdans cette conversation\b",
            ]
            return any(re.search(p, s) for p in patterns)

        if is_chat_memory_question(msg_lc):
            user_msgs = [
                m.get("content", "").strip()
                for m in state.get("messages", [])
                if m.get("role") == "user" and (m.get("content") or "").strip()
            ]
            if user_msgs and user_msgs[-1].strip().lower() == message.strip().lower():
                user_msgs = user_msgs[:-1]

            if user_msgs:
                first_q = user_msgs[0]
                answer = f'Your first question in this chat was: "{first_q}"'
            else:
                answer = "I don't have any earlier question stored in this chat yet."

            state["messages"].append({"role": "assistant", "content": answer, "ts": _now_iso()})
            return {"answer": answer, "user_id": user_id, "sources": []}

        summary = (state["summary"] or "").strip()
        recent = _trim_messages_to_budget(state["messages"], budget_chars=CHAT_CONTEXT_CHAR_BUDGET)

        chunks, uniq_sources = retrieve_chunks_with_sources(
            question=message,
            access_folders=access_folders,
            k_per_doc=6,
            max_total=12,
            chunk_max_chars=1200,
            debug=debug,
        )
        has_docs = bool(chunks)
        docs_block = "\n---\n".join(chunks) if chunks else "(none found for this question)"

        system_rules = (
            "You are the ShipERP assistant.\n"
            "You must answer FIRST using the documentation excerpts provided below.\n"
            "If the documentation contains relevant information, base your answer strictly on it.\n\n"
            "If the documentation does NOT contain the answer, say explicitly:\n"
            '"The documentation does not mention this, but here is what I know from general knowledge:"\n\n'
            "Only then, provide a general-knowledge answer.\n"
            "Do not mix documentation-based information and general knowledge in the same sentence.\n"
            "Be practical and sufficiently detailed to be useful.\n"
            "Do NOT include any 'Sources' section in your answer."
        )

        messages = [{"role": "system", "content": system_rules}]
        if summary:
            messages.append({"role": "system", "content": f"Conversation memory (summary):\n{summary}"})
        messages.append({"role": "system", "content": f"Documentation excerpts:\n{docs_block}"})
        messages.extend(recent)

        answer = chat_completion(model=CHAT_MODEL, messages=messages, max_completion_tokens=900)

        if answer.lower().startswith(FALLBACK_MARKER):
            final_answer = answer
            out_sources = []
        else:
            out_sources = uniq_sources if has_docs else []
            if out_sources:
                final_answer = answer.rstrip() + "\n\nSources: " + ", ".join(out_sources)
            else:
                final_answer = answer

        state["messages"].append({"role": "assistant", "content": final_answer, "ts": _now_iso()})

        if len(state["messages"]) > 2 * CHAT_SUMMARY_TRIGGER:
            state["messages"] = state["messages"][-CHAT_SUMMARY_TRIGGER:]

        payload: Dict[str, Any] = {"answer": final_answer, "user_id": user_id, "sources": out_sources}
        if debug:
            payload["has_docs"] = has_docs
            payload["model"] = CHAT_MODEL

        return payload

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui(request: Request):
    email = (request.session.get("user_email") or "").strip().lower()
    if not get_user_tier(email):
        return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)

    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Conversation Bot (Per User)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
    textarea, button { font-size: 14px; padding: 10px; }
    textarea { width: 100%; height: 90px; }
    #chat { border: 1px solid #ddd; padding: 12px; border-radius: 8px; height: 420px; overflow: auto; background: #fafafa; }
    .msg { margin: 10px 0; }
    .user { font-weight: bold; }
    .assistant { font-weight: bold; }
    .bubble { padding: 10px; border-radius: 10px; display: inline-block; max-width: 90%; white-space: pre-wrap; }
    .b-user { background: #e8f0ff; }
    .b-assistant { background: #e9ffe8; }
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>

<h2>Conversation Bot (per user)</h2>
<p class="muted">Mémoire en RAM (reset au restart). User: <b>__USER_ID__</b></p>

<div id="chat"></div>

<div style="margin-top: 12px;">
  <textarea id="msg" placeholder="Tape ton message..."></textarea>
  <div style="margin-top: 10px;">
    <button onclick="send()">Send</button>
    <form method="post" action="/logout" style="display:inline;">
      <button type="submit">Logout</button>
    </form>
  </div>
</div>

<script>
  const USER_ID = "__USER_ID__";
  const chatEl = document.getElementById('chat');
  const msgEl = document.getElementById('msg');

  function append(role, text) {
    const div = document.createElement('div');
    div.className = 'msg';
    const who = document.createElement('div');
    who.className = role === 'user' ? 'user' : 'assistant';
    who.textContent = role === 'user' ? 'You' : 'Bot';
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (role === 'user' ? 'b-user' : 'b-assistant');
    bubble.textContent = text;
    div.appendChild(who);
    div.appendChild(bubble);
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function newChat() {
    chatEl.innerHTML = '';
    msgEl.value = '';
    append('assistant', 'Nouvelle conversation. Envoie un message.');
  }

  async function send() {
    const message = (msgEl.value || '').trim();
    if (!message) return;

    append('user', message);
    msgEl.value = '';

    const form = new FormData();
    form.append('message', message);

    try {
      const res = await fetch('/chat-api', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Request failed');
      append('assistant', data.answer || '(no answer)');
    } catch (e) {
      append('assistant', 'ERROR: ' + e.message);
    }
  }

  newChat();
</script>

</body>
</html>
"""
    return HTMLResponse(html.replace("__USER_ID__", email))


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
            timeout=10,
        )
    except Exception as e:
        print("[DEBUG][SLACK] error posting follow-up:", e)


def process_slack_question(question: str, response_url: str):
    user_email = "default@erp-is.com"  # internal identity for Slack
    try:
        result = ask_question(question=question, user_email=user_email, debug=False)
        text = result.get("answer") or result.get("error") or "No answer."
    except Exception as e:
        text = f"Error while processing: {str(e)}"

    if response_url:
        post_to_slack(response_url, text)


@app.post("/ask-from-slack")
async def ask_from_slack(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    question = (form.get("text") or "").strip()
    response_url = form.get("response_url")

    if response_url:
        background_tasks.add_task(process_slack_question, question, response_url)

    return {"response_type": "ephemeral", "text": "Got it, I’m generating an answer…"}


# =========================
# Startup sync
# =========================
@app.on_event("startup")
async def startup_event():
    print("Startup complete (sync disabled temporarily).")



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
