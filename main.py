import os
import json
import glob
import asyncio
import datetime
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any, Tuple

import requests
import pdfplumber
import docx
import faiss
import numpy as np
from dotenv import load_dotenv
from msal import ConfidentialClientApplication

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sow_estimator import router as estimator_router


# =========================
# Config
# =========================
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DRIVE_ID = "b!rsvKwTOMNUeCRjsRDMZh-kprgBi3tc1LiWVKbiOrmtWWapTcFH-5QLtKqb12SEmT"
FOLDER_PATH = "AI"

DESTINATION_FOLDER = "documents"
os.makedirs(DESTINATION_FOLDER, exist_ok=True)
HISTORY_LOG = os.path.join(DESTINATION_FOLDER, "sync_log.csv")

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

SCOPES = ["https://graph.microsoft.com/.default"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

INTERNAL_USER_FILE = "internal_users.json"


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
    """
    Extracts text from pdf/docx/txt.
    """
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            with pdfplumber.open(tmp.name) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)

    if ext == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            doc = docx.Document(tmp.name)
            return "\n".join(p.text for p in doc.paragraphs)

    if ext == "txt":
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return ""


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple chunker for docs (not Slack export).
    """
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
    """
    Safe embedding:
    - filters invalid entries
    - truncates long text
    - never crashes sync: skips failing batches
    """
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
        batch = clean_texts[i:i + batch_size]
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
    """
    Loads all {doc}.json + {doc}.index pairs in a folder once, then caches them.
    Returns: List[(chunks: List[str], index: faiss.Index)]
    """
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


def clear_index_cache():
    try:
        load_folder_indexes.cache_clear()
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

        # keep original file locally (optional)
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
# Internal users
# =========================
if os.path.exists(INTERNAL_USER_FILE):
    with open(INTERNAL_USER_FILE, "r", encoding="utf-8") as f:
        INTERNAL_USERS = json.load(f)
else:
    INTERNAL_USERS = {}


# =========================
# FastAPI
# =========================
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
# One-shot QA (/ask) - docs-first + fallback
# =========================
@app.post("/ask")
def ask_question(question: str = Form(...), user_email: str = Form(...)):
    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        # Access control
        access_folders = ["documents/public"]
        if user_email and user_email.endswith("@erp-is.com"):
            access_folders.append("documents/internal")
        elif INTERNAL_USERS.get(user_email):
            access_folders.append("documents/internal")

        # Embed
        qvecs = embed_texts([question])
        if not qvecs:
            return {"answer": "Invalid or empty question."}
        qvec = qvecs[0]

        # Retrieve
        scored: List[Tuple[float, str]] = []
        for folder in access_folders:
            for chunks, index in load_folder_indexes(folder):
                D, I = index.search(np.array([qvec]).astype("float32"), k=8)
                for dist, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(chunks):
                        scored.append((float(dist), chunks[idx]))

        if not scored:
            top_chunks = []
        else:
            scored.sort(key=lambda x: x[0])
            seen = set()
            top_chunks = []
            for _, c in scored:
                c = (c or "").strip()
                if not c or c in seen:
                    continue
                seen.add(c)
                top_chunks.append(c)
                if len(top_chunks) >= 12:
                    break

        system_rules = (
            "You are the ShipERP assistant.\n"
            "You must answer FIRST using the documentation excerpts provided below.\n"
            "If the documentation contains relevant information, base your answer strictly on it.\n\n"
            "If the documentation does NOT contain the answer, say explicitly:\n"
            "\"The documentation does not mention this, but here is what I know from general knowledge:\"\n\n"
            "Only then, provide a concise general-knowledge answer.\n"
            "Do not mix documentation-based information and general knowledge in the same sentence."
        )

        docs_block = "\n---\n".join(top_chunks) if top_chunks else "(none found for this question)"
        user_prompt = f"""Documentation excerpts:
{docs_block}

Question: {question}
"""

        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return {"answer": resp.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}


# =========================
# Conversation (per-user) + RAG
# =========================
CHAT_MODEL = "gpt-4"          # or "gpt-4o-mini" if you switch later
CHAT_MAX_TURNS = 12
CHAT_SUMMARY_TRIGGER = 18
CHAT_CONTEXT_CHAR_BUDGET = 14000

# In-memory store (resets on restart)
CHAT_STORE: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _trim_messages_to_budget(messages: List[dict], budget_chars: int) -> List[dict]:
    """
    Keep newest messages within a rough char budget.
    """
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
    """
    Summarize older turns into 'summary' if too many messages.
    """
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

    import openai
    openai.api_key = OPENAI_API_KEY

    summary_prompt = f"""You are maintaining a running summary of a chat.

Current summary (may be empty):
{state['summary']}

New dialogue to summarize:
{old_text}

Update the summary. Keep it factual and concise. Preserve decisions, constraints, names, and open questions.
"""

    try:
        resp = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You summarize chat history into a concise running memory."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.2,
        )
        state["summary"] = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[CHAT] summarization failed:", e)

    state["messages"] = keep


def _trim_chars(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


def _trim_list_to_char_budget(items: List[str], budget_chars: int) -> List[str]:
    out = []
    total = 0
    for it in items:
        it = (it or "").strip()
        if not it:
            continue
        if total + len(it) + 2 > budget_chars:
            break
        out.append(it)
        total += len(it) + 2
    return out


def retrieve_chunks(question: str, access_folders: List[str], k_per_doc: int = 6, max_total: int = 12) -> List[str]:
    """
    RAG retrieval from FAISS indexes.
    Returns a list of best chunks across all accessible folders.
    """
    qvecs = embed_texts([question])
    if not qvecs:
        return []
    qvec = qvecs[0]

    scored: List[Tuple[float, str]] = []
    for folder in access_folders:
        for chunks, index in load_folder_indexes(folder):
            D, I = index.search(np.array([qvec]).astype("float32"), k=k_per_doc)
            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(chunks):
                    scored.append((float(dist), chunks[idx]))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0])  # smaller distance = better
    seen = set()
    out: List[str] = []
    for _, c in scored:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= max_total:
            break
    return out


@app.post("/chat-api")
async def chat_api(
    user_id: str = Form(...),
    message: str = Form(...),
    debug: bool = Form(False),  # optional: /chat-api accepts debug=true
):
    """
    Stateful chat per user_id + docs RAG.

    Docs-first rule:
    - Answer using docs excerpts if they contain answer.
    - Otherwise: say exact fallback sentence then general knowledge.

    Adds "Sources" in the response (doc file basenames used for excerpts).
    Adds optional debug logs in Render when debug=true.
    """
    try:
        import openai
        import numpy as np

        user_id = (user_id or "").strip()
        message = (message or "").strip()

        if not user_id:
            return JSONResponse({"error": "Missing user_id"}, status_code=400)
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        # ---- tiny local helpers (avoid missing globals) ----
        def _trim_chars_local(s: str, n: int) -> str:
            s = (s or "").strip()
            if len(s) <= n:
                return s
            return s[:n].rstrip() + "…"

        def _trim_list_to_char_budget_local(items: list[str], budget_chars: int) -> list[str]:
            out, total = [], 0
            for it in items:
                it = (it or "").strip()
                if not it:
                    continue
                add = len(it) + 5
                if out and (total + add) > budget_chars:
                    break
                out.append(it)
                total += add
            return out

        # ---- Access control (same as /ask) ----
        access_folders = ["documents/public"]
        if user_id.lower().endswith("@erp-is.com"):
            access_folders.append("documents/internal")
        elif INTERNAL_USERS.get(user_id):
            access_folders.append("documents/internal")

        # ---- Conversation state ----
        state = _get_user_state(user_id)
        state["messages"].append({"role": "user", "content": message, "ts": _now_iso()})

        _summarize_if_needed(user_id)

        summary = (state["summary"] or "").strip()
        recent = _trim_messages_to_budget(state["messages"], budget_chars=CHAT_CONTEXT_CHAR_BUDGET)

        # ---- RAG retrieval WITH sources (doc name) ----
        # We do it inline so we can attach doc basenames without changing your global retrieve_chunks().
        qvecs = embed_texts([message])
        if not qvecs:
            return JSONResponse({"error": "Invalid question (not embeddable)."}, status_code=400)
        qvec = qvecs[0]

        scored: list[tuple[float, str, str]] = []  # (dist, chunk, doc_name)

        for folder in access_folders:
            # We need doc names => scan json files to infer base names.
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
                        print("[CHAT][DEBUG] Failed loading doc:", json_path, "err:", e)
                    continue

                # search
                D, I = index.search(np.array([qvec]).astype("float32"), k=6)
                for dist, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(chunks_list):
                        c = chunks_list[idx]
                        if c:
                            scored.append((float(dist), str(c), base))

        scored.sort(key=lambda x: x[0])  # smaller = better

        # dedupe chunks while keeping best + keep sources
        seen_chunks = set()
        chosen_chunks: list[str] = []
        chosen_sources: list[str] = []  # parallel to chosen_chunks

        for dist, c, doc_name in scored:
            c = (c or "").strip()
            if not c:
                continue
            if c in seen_chunks:
                continue
            seen_chunks.add(c)

            chosen_chunks.append(_trim_chars_local(c, 1200))
            chosen_sources.append(doc_name)

            if len(chosen_chunks) >= 12:
                break

        chosen_chunks = _trim_list_to_char_budget_local(chosen_chunks, budget_chars=9000)

        # align sources length if trimming cut off (safe)
        chosen_sources = chosen_sources[: len(chosen_chunks)]

        # ---- Debug logs (Render) ----
        if debug:
            print("[CHAT][DEBUG] user_id:", user_id)
            print("[CHAT][DEBUG] message:", message)
            print("[CHAT][DEBUG] access_folders:", access_folders)
            print("[CHAT][DEBUG] retrieved_chunks:", len(chosen_chunks))
            for i, (c, src) in enumerate(zip(chosen_chunks[:8], chosen_sources[:8]), start=1):
                preview = c.replace("\n", " ")[:260]
                print(f"[CHAT][DEBUG] chunk#{i} src={src} :: {preview}")

        # ---- Build docs block + sources list ----
        if chosen_chunks:
            docs_block = "\n---\n".join(chosen_chunks)
            # unique sources in display order
            uniq_sources = []
            seen = set()
            for s in chosen_sources:
                if s and s not in seen:
                    seen.add(s)
                    uniq_sources.append(s)
        else:
            docs_block = "(none found for this question)"
            uniq_sources = []

        system_rules = (
            "You are the ShipERP assistant.\n"
            "You must answer FIRST using the documentation excerpts provided below.\n"
            "If the documentation contains relevant information, base your answer strictly on it.\n\n"
            "If the documentation does NOT contain the answer, say explicitly:\n"
            "\"The documentation does not mention this, but here is what I know from general knowledge:\"\n\n"
            "Only then, provide a general-knowledge answer.\n"
            "Do not mix documentation-based information and general knowledge in the same sentence.\n"
            "Be practical and sufficiently detailed to be useful.\n"
            "At the end of your answer, include a 'Sources:' section listing the document names used."
        )

        messages = [{"role": "system", "content": system_rules}]
        if summary:
            messages.append({"role": "system", "content": f"Conversation memory (summary):\n{summary}"})

        messages.append({"role": "system", "content": f"Documentation excerpts:\n{docs_block}"})
        messages.extend(recent)

        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=850,  # helps get less "short" answers
        )

        answer = (resp.choices[0].message.content or "").strip()

        # If the model forgot to add sources, we append them (so users always get them).
        # (We still keep the instruction in system_rules for nicer formatting.)
        sources_block = "Sources: " + (", ".join(uniq_sources) if uniq_sources else "(none)")
        if "sources:" not in answer.lower():
            answer = answer.rstrip() + "\n\n" + sources_block

        state["messages"].append({"role": "assistant", "content": answer, "ts": _now_iso()})

        if len(state["messages"]) > 2 * CHAT_SUMMARY_TRIGGER:
            state["messages"] = state["messages"][-CHAT_SUMMARY_TRIGGER:]

        return {"answer": answer, "user_id": user_id, "sources": uniq_sources}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Conversation Bot (Per User)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
    .row { display: flex; gap: 12px; margin-bottom: 12px; }
    input, textarea, button { font-size: 14px; padding: 10px; }
    input { flex: 1; }
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
<p class="muted">Cette page est séparée du bot Slack/one-shot. La mémoire est en RAM (reset au restart).</p>

<div class="row">
  <input id="user_id" placeholder="user_id (ex: email ou username)" />
  <button onclick="newChat()">New chat</button>
</div>

<div id="chat"></div>

<div style="margin-top: 12px;">
  <textarea id="msg" placeholder="Tape ton message..."></textarea>
  <div class="row">
    <button onclick="send()">Send</button>
  </div>
</div>

<script>
  const chatEl = document.getElementById('chat');
  const userIdEl = document.getElementById('user_id');
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
    const user_id = (userIdEl.value || '').trim();
    const message = (msgEl.value || '').trim();
    if (!user_id) return alert('Please set user_id');
    if (!message) return;

    append('user', message);
    msgEl.value = '';

    const form = new FormData();
    form.append('user_id', user_id);
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
    # internal identity for Slack
    user_email = "default@erp-is.com"
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
