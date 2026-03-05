# main.py
import os
import json
import glob
import asyncio
import datetime
import tempfile
import sqlite3
import threading
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse

import bcrypt
import requests
import pdfplumber
import docx
import faiss
import numpy as np

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_303_SEE_OTHER

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from dotenv import load_dotenv
from msal import ConfidentialClientApplication

from openai import OpenAI
client = OpenAI()

from sow_estimator import router as estimator_router


# =========================
# Config
# =========================
load_dotenv()

CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# Models
MODEL_MINI = os.getenv("OPENAI_MODEL_MINI", "gpt-5-mini")
MODEL_BIG = os.getenv("OPENAI_MODEL_BIG", "gpt-5.2")

# SharePoint
DRIVE_ID = os.getenv(
    "SHAREPOINT_DRIVE_ID",
    "b!rsvKwTOMNUeCRjsRDMZh-kprgBi3tc1LiWVKbiOrmtWWapTcFH-5QLtKqb12SEmT",
)
FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH", "AI")  # root folder in drive

# Storage
BASE_STORAGE = "/data"
DESTINATION_FOLDER = os.path.join(BASE_STORAGE, "documents")  # /data/documents/{tier}/...
INDEXES_FOLDER = os.path.join(BASE_STORAGE, "indexes")        # /data/indexes/{tier}/faiss.index + meta.db
LOGS_FOLDER = os.path.join(BASE_STORAGE, "logs")

os.makedirs(DESTINATION_FOLDER, exist_ok=True)
os.makedirs(INDEXES_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

HISTORY_LOG = os.path.join(LOGS_FOLDER, "sync_log.csv")

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

# Azure auth
SCOPES = ["https://graph.microsoft.com/.default"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

# Auth users
PUBLIC_USER_FILE = "public_users.json"
INTERNAL_USER_FILE = "internal_users.json"

# Frontend allowlist
FRONTEND_ORIGINS = [
    "https://docqa-frontend-git-main-antoine-lauwens-projects.vercel.app",
    # later: "https://portal.shiperp.com",
]

FALLBACK_MARKER = "the documentation does not mention this, but here is what i know from general knowledge:"


# =========================
# Utils: users
# =========================
def _load_users(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}

INTERNAL_USERS = _load_users(INTERNAL_USER_FILE)  # email -> True OR {"password_hash": "..."}
PUBLIC_USERS = _load_users(PUBLIC_USER_FILE)      # email -> True OR {"password_hash": "..."}

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


# =========================
# OpenAI client + wrappers (SDK v2.x)
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

def chat_completion(model: str, messages: List[dict], max_completion_tokens: int = 700) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    clean_texts: List[str] = []
    for t in texts:
        if not isinstance(t, str):
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
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        except Exception as e:
            print("[EMBEDDING ERROR] batch skipped:", repr(e))
    return vectors


# =========================
# Consolidated index loading (public/internal)
# =========================
_FAISS_LOCK = threading.Lock()

@lru_cache(maxsize=4)
def load_consolidated_index(tier: str):
    """
    Load FAISS + sqlite meta for a tier ("public" / "internal") from /data/indexes/{tier}.
    Returns (faiss_index, sqlite_db_path).
    """
    tier = (tier or "").strip().lower()
    if tier not in ("public", "internal"):
        raise ValueError("tier must be public or internal")

    idx_dir = Path(INDEXES_FOLDER) / tier
    faiss_path = idx_dir / "faiss.index"
    db_path = idx_dir / "meta.db"

    if not faiss_path.exists() or not db_path.exists():
        raise FileNotFoundError(f"Missing consolidated index for {tier}: {faiss_path} / {db_path}")

    index = faiss.read_index(str(faiss_path))
    return index, str(db_path)

def fetch_chunks_from_meta(db_path: str, ids_1based: List[int]) -> List[Tuple[str, str]]:
    """
    Returns list of (file, chunk) for given sqlite row ids (1-based).
    Preserves input order.
    """
    if not ids_1based:
        return []
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        out: List[Tuple[str, str]] = []
        for rid in ids_1based:
            c.execute("SELECT file, chunk FROM chunks WHERE id = ?", (rid,))
            row = c.fetchone()
            if row and row[1]:
                out.append((row[0] or "", row[1] or ""))
        return out
    finally:
        conn.close()

def search_consolidated(tier: str, qvec: List[float], k: int = 12) -> List[Tuple[float, str, str]]:
    """
    Returns list of (dist, chunk, source_filebasename) from consolidated index.
    """
    index, db_path = load_consolidated_index(tier)

    with _FAISS_LOCK:
        D, I = index.search(np.array([qvec]).astype("float32"), k=k)

    # FAISS ids are 0-based; sqlite AUTOINCREMENT ids are 1-based
    ids_1based: List[int] = []
    dists: List[float] = []
    for dist, idx0 in zip(D[0], I[0]):
        if idx0 < 0:
            continue
        ids_1based.append(int(idx0) + 1)
        dists.append(float(dist))

    rows = fetch_chunks_from_meta(db_path, ids_1based)
    scored: List[Tuple[float, str, str]] = []
    for dist, (file_path, chunk) in zip(dists, rows):
        src = os.path.basename(file_path) if file_path else "unknown"
        scored.append((dist, (chunk or "").strip(), src))
    return scored


# =========================
# SSO tokens
# =========================
SSO_SECRET = os.getenv("SSO_SECRET", "")
if not SSO_SECRET:
    raise RuntimeError("Missing SSO_SECRET env var")

_sso = URLSafeTimedSerializer(SSO_SECRET, salt="docqa-sso-v1")

def issue_sso_token(email: str) -> str:
    email = (email or "").strip().lower()
    tier = get_user_tier(email)
    if not tier:
        raise HTTPException(status_code=403, detail="Access denied")
    return _sso.dumps({"email": email})

def verify_sso_token(token: str, max_age_seconds: int = 300) -> Dict[str, Any]:
    try:
        data = _sso.loads(token, max_age=max_age_seconds)
        if not isinstance(data, dict) or not data.get("email"):
            raise HTTPException(status_code=401, detail="Invalid token")
        return data
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Token expired")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid token")

def _is_allowed_next(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.scheme != "https":
            return False
        origin = f"{u.scheme}://{u.netloc}"
        return origin in FRONTEND_ORIGINS
    except Exception:
        return False


# =========================
# SharePoint auth + sync (download only)
# =========================
def authenticate_graph() -> str:
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise Exception(f"Authentication failed: {result}")
    return result["access_token"]

def log_sync_activity(filename: str, user_name: str, user_email: str):
    ts = datetime.datetime.utcnow().isoformat()
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts},{filename},{user_name},{user_email}\n")

def _graph_children_by_path(headers: dict, drive_id: str, path: str) -> List[dict]:
    # path is like "AI/public/PPT_PDF"
    encoded = path.replace(" ", "%20")
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{encoded}:/children"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("value", []) or []

def sync_sharepoint_download_only(
    tier: str,
    sp_subpath: str,
    dest_root: str,
    allowed_exts: Optional[List[str]] = None,
    max_files: Optional[int] = None,
):
    """
    Recursively downloads allowed file types from SharePoint folder path:
      {FOLDER_PATH}/{sp_subpath}
    into:
      {dest_root}/{tier}/... (preserves subfolders)

    This function does NOT build embeddings. You run build_index.py separately.
    """
    tier = (tier or "").strip().lower()
    if tier not in ("public", "internal"):
        raise ValueError("tier must be public or internal")

    allowed_exts = allowed_exts or ["pdf", "docx", "txt"]
    allowed_exts = [e.lower().lstrip(".") for e in allowed_exts]

    access_token = authenticate_graph()
    headers = {"Authorization": f"Bearer {access_token}"}

    base_sp_path = f"{FOLDER_PATH}/{sp_subpath}".strip("/")

    dest_tier_root = Path(dest_root) / tier
    dest_tier_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    def walk(sp_path: str, rel_dest: Path):
        nonlocal downloaded
        if max_files is not None and downloaded >= max_files:
            return

        items = _graph_children_by_path(headers, DRIVE_ID, sp_path)

        for item in items:
            if max_files is not None and downloaded >= max_files:
                return

            name = item.get("name") or ""
            if not name:
                continue

            is_folder = "folder" in item
            if is_folder:
                walk(f"{sp_path}/{name}", rel_dest / name)
                continue

            ext = name.lower().split(".")[-1]
            if ext not in allowed_exts:
                continue

            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                continue

            dest_dir = dest_tier_root / rel_dest
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / name
            if dest_path.exists() and dest_path.stat().st_size > 0:
                continue

            print(f"Downloading: {sp_path}/{name}")
            file_data = requests.get(download_url, timeout=180)
            file_data.raise_for_status()
            dest_path.write_bytes(file_data.content)

            user_info = item.get("createdBy", {}).get("user", {}) or {}
            user_name = user_info.get("displayName", "Unknown")
            user_email = user_info.get("email", "Unknown")
            log_sync_activity(f"{sp_path}/{name}", user_name, user_email)

            downloaded += 1

    walk(base_sp_path, Path("."))
    print(f"✅ Download sync complete for tier={tier}. Downloaded {downloaded} file(s).")


# =========================
# Upload parsing + chunking (uploads only)
# =========================
def extract_text(filename: str, content: bytes) -> str:
    ext = (filename or "").lower().split(".")[-1]

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
    text = (text or "").strip()
    if not text:
        return []
    sentences = text.split(". ")
    chunks: List[str] = []
    buf = ""
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

def _safe_email_folder(email: str) -> str:
    return (
        (email or "")
        .replace("@", "_at_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


# =========================
# Retrieval: consolidated (public/internal) + uploads per-doc
# =========================
def retrieve_chunks_with_sources(
    question: str,
    tier: str,
    user_upload_folder: str,
    k_public_internal: int = 18,
    k_per_doc_upload: int = 6,
    max_total: int = 12,
    chunk_max_chars: int = 1200,
    debug: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Hybrid retrieval:
    - consolidated index for tier ("public"/"internal") from /data/indexes/{tier}
    - per-user uploads still per-doc index (*.json + *.index) under /data/documents/uploads/{safe_email}
    """
    question = (question or "").strip()
    if not question:
        return [], []

    qvecs = embed_texts([question])
    if not qvecs:
        return [], []
    qvec = qvecs[0]

    scored: List[Tuple[float, str, str]] = []  # (dist, chunk, source)

    # 1) consolidated public/internal
    try:
        scored.extend(search_consolidated("public", qvec, k=k_public_internal))
        if tier == "internal":
            scored.extend(search_consolidated("internal", qvec, k=k_public_internal))
    except Exception as e:
        if debug:
            print("[RAG][DEBUG] consolidated search error:", e)

    # 2) user uploads (per-doc)
    try:
        folder = user_upload_folder
        json_paths = glob.glob(f"{folder}/*.json")
        for json_path in json_paths:
            base = os.path.splitext(os.path.basename(json_path))[0]
            index_path = os.path.join(folder, f"{base}.index")
            if not os.path.exists(index_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                chunks_list = json.load(f)

            idx = faiss.read_index(index_path)
            with _FAISS_LOCK:
                D, I = idx.search(np.array([qvec]).astype("float32"), k=k_per_doc_upload)

            for dist, ci in zip(D[0], I[0]):
                if 0 <= ci < len(chunks_list):
                    c = (chunks_list[ci] or "").strip()
                    if c:
                        scored.append((float(dist), c, f"upload:{base}"))
    except Exception as e:
        if debug:
            print("[RAG][DEBUG] upload search error:", e)

    if not scored:
        return [], []

    scored.sort(key=lambda x: x[0])

    chunks: List[str] = []
    sources: List[str] = []
    seen_chunks = set()
    seen_src = set()

    for dist, c, src in scored:
        if not c or c in seen_chunks:
            continue
        seen_chunks.add(c)

        if len(c) > chunk_max_chars:
            c = c[:chunk_max_chars].rstrip() + "…"

        chunks.append(c)

        if src and src not in seen_src:
            seen_src.add(src)
            sources.append(src)

        if len(chunks) >= max_total:
            break

    if debug:
        print("[RAG][DEBUG] tier:", tier, "chunks:", len(chunks), "sources:", sources)

    return chunks, sources


# =========================
# FastAPI app
# =========================
app = FastAPI()

SESSION_SECRET = os.getenv("SESSION_SECRET", "")
if not SESSION_SECRET:
    raise RuntimeError("Missing SESSION_SECRET env var")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="none",
    https_only=True,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(estimator_router)


# =========================
# Auth endpoints
# =========================
@app.post("/auth/login")
def auth_login(email: str = Form(...), password: str = Form(...)):
    email_lc = (email or "").strip().lower()

    if not get_user_tier(email_lc):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not verify_password(email_lc, password or ""):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)

    token = issue_sso_token(email_lc)
    return {"token": token, "email": email_lc}

@app.get("/sso")
def sso_entry(
    request: Request,
    token: str,
    next_url: str = Query(..., alias="next"),
):
    data = verify_sso_token(token, max_age_seconds=300)
    email = (data.get("email") or "").strip().lower()

    if not get_user_tier(email):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not next_url or not _is_allowed_next(next_url):
        return JSONResponse({"error": "Invalid next URL"}, status_code=400)

    request.session["user_email"] = email
    return RedirectResponse(url=next_url, status_code=HTTP_303_SEE_OTHER)

@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@app.get("/me")
def me(request: Request):
    email = (request.session.get("user_email") or "").strip().lower()
    tier = get_user_tier(email)
    if not tier:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return {"email": email, "tier": tier}

@app.get("/")
def root():
    return {"status": "ok", "service": "docqa-api"}


# =========================
# Upload (private per-user)
# =========================
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    email = (request.session.get("user_email") or "").strip().lower()
    tier = get_user_tier(email)
    if not tier:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        filename = file.filename or "document"
        document_name = os.path.splitext(filename)[0] or "document"

        content = await file.read()
        text = extract_text(filename, content)
        chunks = chunk_text(text)
        if not chunks:
            return JSONResponse({"error": "No text extracted from uploaded file."}, status_code=400)

        vectors = embed_texts(chunks)
        if not vectors:
            return JSONResponse({"error": "No valid text to embed in this document."}, status_code=400)

        safe_email = _safe_email_folder(email)
        user_upload_folder = os.path.join(DESTINATION_FOLDER, "uploads", safe_email)
        os.makedirs(user_upload_folder, exist_ok=True)

        json_path = os.path.join(user_upload_folder, f"{document_name}.json")
        index_path = os.path.join(user_upload_folder, f"{document_name}.index")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        idx = faiss.IndexFlatL2(VECTOR_DIM)
        idx.add(np.array(vectors).astype("float32"))
        faiss.write_index(idx, index_path)

        return {"message": f"Document '{document_name}' uploaded privately for {email}."}

    except Exception as e:
        print("Upload failed:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Sync now (download-only; indexing is separate)
# =========================
sync_in_progress = False

@app.post("/sync-now")
def trigger_sync(background_tasks: BackgroundTasks):
    """
    Downloads updated docs from SharePoint into /data/documents/{tier}.
    NOTE: Does NOT build /data/indexes/...; run build_index.py separately.
    """
    global sync_in_progress
    if sync_in_progress:
        return {"status": "busy", "message": "Sync already in progress."}

    sync_in_progress = True

    # you can override these via env:
    SP_PUBLIC_SUBPATH = os.getenv("SP_PUBLIC_SUBPATH", "public")
    SP_INTERNAL_SUBPATH = os.getenv("SP_INTERNAL_SUBPATH", "internal")

    def run_and_release():
        global sync_in_progress
        try:
            # public
            sync_sharepoint_download_only(
                tier="public",
                sp_subpath=SP_PUBLIC_SUBPATH,
                dest_root=DESTINATION_FOLDER,
                allowed_exts=["pdf", "docx", "txt"],
            )
            # internal
            sync_sharepoint_download_only(
                tier="internal",
                sp_subpath=SP_INTERNAL_SUBPATH,
                dest_root=DESTINATION_FOLDER,
                allowed_exts=["pdf", "docx", "txt"],
            )
        except Exception as e:
            print("[SYNC ERROR]", str(e))
        finally:
            sync_in_progress = False

    background_tasks.add_task(run_and_release)
    return {"status": "started", "message": "SharePoint download sync started in background."}


# =========================
# One-shot QA (/ask)
# =========================
@app.post("/ask")
def ask_question(
    question: str = Form(...),
    user_email: str = Form(...),
    debug: bool = Form(False),
):
    try:
        question = (question or "").strip()
        email_lc = (user_email or "").strip().lower()

        tier = get_user_tier(email_lc)
        if not tier:
            return JSONResponse({"error": "Access denied"}, status_code=403)

        safe_email = _safe_email_folder(email_lc)
        user_upload_folder = os.path.join(DESTINATION_FOLDER, "uploads", safe_email)
        os.makedirs(user_upload_folder, exist_ok=True)

        chunks, sources = retrieve_chunks_with_sources(
            question=question,
            tier=tier,
            user_upload_folder=user_upload_folder,
            k_public_internal=18,
            k_per_doc_upload=6,
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

        answer = chat_completion(model=MODEL_BIG, messages=messages, max_completion_tokens=700)

        if answer.lower().startswith(FALLBACK_MARKER):
            return {"answer": answer, "sources": []}

        out_sources = sources if has_docs else []
        final_answer = answer.rstrip() + (("\n\nSources: " + ", ".join(out_sources)) if out_sources else "")

        payload: Dict[str, Any] = {"answer": final_answer, "sources": out_sources}
        if debug:
            payload["has_docs"] = has_docs
            payload["model"] = MODEL_BIG
            payload["tier"] = tier
            payload["user_upload_folder"] = user_upload_folder

        return payload

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Conversation (per-user) + RAG
# =========================
CHAT_MODEL = MODEL_BIG
CHAT_MAX_TURNS = 12
CHAT_SUMMARY_TRIGGER = 18
CHAT_CONTEXT_CHAR_BUDGET = 14000

CHAT_STORE: Dict[str, Dict[str, Any]] = {}  # in-memory, resets on restart

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

    message = (message or "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    safe_email = _safe_email_folder(email)
    user_upload_folder = os.path.join(DESTINATION_FOLDER, "uploads", safe_email)
    os.makedirs(user_upload_folder, exist_ok=True)

    try:
        import re

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
            tier=tier,
            user_upload_folder=user_upload_folder,
            k_public_internal=18,
            k_per_doc_upload=6,
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
            final_answer = answer.rstrip() + (("\n\nSources: " + ", ".join(out_sources)) if out_sources else "")

        state["messages"].append({"role": "assistant", "content": final_answer, "ts": _now_iso()})

        if len(state["messages"]) > 2 * CHAT_SUMMARY_TRIGGER:
            state["messages"] = state["messages"][-CHAT_SUMMARY_TRIGGER:]

        payload: Dict[str, Any] = {"answer": final_answer, "user_id": user_id, "sources": out_sources}
        if debug:
            payload["has_docs"] = has_docs
            payload["model"] = CHAT_MODEL
            payload["tier"] = tier

        return payload

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Admin pages
# =========================
def require_internal(request: Request) -> str:
    email = (request.session.get("user_email") or "").strip().lower()
    if get_user_tier(email) != "internal":
        raise HTTPException(status_code=403, detail="Forbidden")
    return email

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    require_internal(request)
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
def add_internal_user(request: Request, email: str = Form(...)):
    require_internal(request)
    email = (email or "").strip().lower()
    INTERNAL_USERS[email] = True
    with open(INTERNAL_USER_FILE, "w", encoding="utf-8") as f:
        json.dump(INTERNAL_USERS, f, indent=2)
    return HTMLResponse(f"<p>{email} added as internal user. <a href='/admin'>Back</a></p>")

@app.post("/admin/remove")
def remove_internal_user(request: Request, email: str = Form(...)):
    require_internal(request)
    email = (email or "").strip().lower()
    if email in INTERNAL_USERS:
        del INTERNAL_USERS[email]
        with open(INTERNAL_USER_FILE, "w", encoding="utf-8") as f:
            json.dump(INTERNAL_USERS, f, indent=2)
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
# Startup
# =========================
@app.on_event("startup")
async def startup_event():
    # Default: disabled (avoids surprise memory spikes / restarts)
    if os.getenv("DISABLE_STARTUP_SYNC", "1") == "1":
        print("Startup sync disabled.")
        return

    async def _run():
        try:
            print("Running SharePoint download sync on startup (background)...")
            # Optional: you can set SP_PUBLIC_SUBPATH/SP_INTERNAL_SUBPATH via env
            sp_public = os.getenv("SP_PUBLIC_SUBPATH", "public")
            sp_internal = os.getenv("SP_INTERNAL_SUBPATH", "internal")

            await asyncio.to_thread(
                sync_sharepoint_download_only,
                "public",
                sp_public,
                DESTINATION_FOLDER,
                ["pdf", "docx", "txt"],
                None,
            )
            await asyncio.to_thread(
                sync_sharepoint_download_only,
                "internal",
                sp_internal,
                DESTINATION_FOLDER,
                ["pdf", "docx", "txt"],
                None,
            )

            print("Startup download sync finished.")
        except Exception as e:
            import traceback
            print("Startup sync failed:", repr(e))
            traceback.print_exc()

    asyncio.create_task(_run())
    print("Startup complete (sync scheduled).")
