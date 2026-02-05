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

# Option B: mini by default + escalate to 5.2 when needed
MODEL_MINI = os.getenv("OPENAI_MODEL_MINI", "gpt-5-mini")
MODEL_BIG = os.getenv("OPENAI_MODEL_BIG", "gpt-5.2")
TRIAGE_CONFIDENCE_THRESHOLD = float(os.getenv("TRIAGE_THRESHOLD", "0.82"))

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
# OpenAI helpers
# =========================
def chat_completion(model: str, messages: List[dict], temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Wrapper compatible with your current OpenAI SDK usage (ChatCompletion).
    """
    import openai

    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def should_escalate_fast(text: str, has_docs: bool) -> bool:
    """
    Cheap deterministic guardrail: forces escalation on sensitive/complex signals.
    Keeps your "low tolerance" behavior even if triage JSON fails.
    """
    t = (text or "").strip().lower()
    if not t:
        return True

    # Long / potentially multi-step
    if len(t) > 700:
        return True

    # Sensitive / precision-critical topics
    keywords = [
        "billing", "invoice", "pricing", "price", "cost", "facturation", "devis",
        "security", "sécurité", "apikey", "api key", "token", "secret", "credential",
        "rgpd", "gdpr", "contract", "contrat", "legal", "juridique",
        "delete", "remove", "drop", "purge", "irreversible", "production"
    ]
    if any(k in t for k in keywords):
        return True

    # If no docs, and user asks for a precise "how/why" → escalate more often
    if not has_docs and any(x in t for x in ["how", "why", "comment", "pourquoi", "explain", "explique"]):
        return True

    return False


def triage_route(user_text: str, mode: str, has_docs: bool, context_hint: str = "") -> dict:
    """
    Triage using MODEL_MINI returning strict JSON.
    If JSON is invalid → safe fallback = escalate.
    """
    user_text = (user_text or "").strip()

    system = (
        "You are a routing classifier for an assistant.\n"
        "Decide if the request can be answered safely and correctly by a small/cheap model "
        "or if it must be escalated to a high-quality model.\n"
        "Return ONLY valid JSON with the exact keys specified.\n\n"
        "Escalate if:\n"
        "- confidence is low\n"
        "- request is multi-step, ambiguous, or requires careful reasoning\n"
        "- request is sensitive (security, credentials, billing, legal, irreversible actions)\n"
        "- answer requires long structured output or high precision\n"
        "- documentation is missing/unclear and the user expects correctness\n"
    )

    payload = {
        "mode": mode,  # "slack" | "web"
        "has_docs": has_docs,
        "context_hint": (context_hint or "")[:1500],
        "user_message": user_text[:5000],
        "output_schema": {
            "route": "mini|escalate",
            "confidence": "0.0-1.0",
            "answer_draft": "string (only if route=mini)",
            "handoff": {
                "summary": "string",
                "key_facts": "array of strings",
                "open_questions": "array of strings"
            }
        }
    }

    raw = chat_completion(
        model=MODEL_MINI,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=450,
    )

    try:
        data = json.loads(raw)
    except Exception:
        return {
            "route": "escalate",
            "confidence": 0.0,
            "answer_draft": "",
            "handoff": {"summary": "", "key_facts": [], "open_questions": []},
        }

    route = (data.get("route") or "").strip().lower()
    conf = data.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    if route not in ("mini", "escalate"):
        route = "escalate"

    data["route"] = route
    data["confidence"] = conf
    if "handoff" not in data or not isinstance(data["handoff"], dict):
        data["handoff"] = {"summary": "", "key_facts": [], "open_questions": []}
    return data


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
    """Simple chunker for docs (not Slack export)."""
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
    """
    Same as load_folder_indexes, but includes doc name (basename without extension).
    Returns: List[(doc_name, chunks, index)]
    """
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
    """
    Returns:
      chunks: list[str] (deduped best chunks)
      sources: list[str] unique doc basenames (deduped, in order of first appearance)
    """
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

    scored.sort(key=lambda x: x[0])  # smaller = better

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
# One-shot QA (/ask) - docs-first + fallback + Option B routing
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
        access_folders = ["documents/public"]
        if user_email and user_email.endswith("@erp-is.com"):
            access_folders.append("documents/internal")
        elif INTERNAL_USERS.get(user_email):
            access_folders.append("documents/internal")

        # Retrieve
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
            "\"The documentation does not mention this, but here is what I know from general knowledge:\"\n\n"
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

        # Option B routing
        fast_force = should_escalate_fast(question, has_docs=has_docs)
        triage = triage_route(
            user_text=question,
            mode="slack_one_shot",
            has_docs=has_docs,
            context_hint="One-shot question. Docs excerpts provided." if has_docs else "One-shot question. No docs found.",
        )

        use_mini = (
            (not fast_force)
            and triage.get("route") == "mini"
            and float(triage.get("confidence", 0.0)) >= TRIAGE_CONFIDENCE_THRESHOLD
        )
        model = MODEL_MINI if use_mini else MODEL_BIG

        answer = chat_completion(model=model, messages=messages, temperature=0.2, max_tokens=700)

        # Sources: only if we actually had docs chunks
        out_sources = sources if has_docs else []
        if has_docs and out_sources:
            answer = answer.rstrip() + "\n\nSources: " + ", ".join(out_sources)

        payload = {"answer": answer, "sources": out_sources}
        if debug:
            payload["routed_model"] = model
            payload["triage"] = triage
            payload["fast_force_escalate"] = fast_force
            payload["threshold"] = TRIAGE_CONFIDENCE_THRESHOLD

        return payload

    except Exception as e:
        return {"error": str(e)}


# =========================
# Conversation (per-user) + RAG + Option B routing
# =========================
CHAT_MAX_TURNS = 12
CHAT_SUMMARY_TRIGGER = 18
CHAT_CONTEXT_CHAR_BUDGET = 14000

# In-memory store (resets on restart)
CHAT_STORE: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _trim_messages_to_budget(messages: List[dict], budget_chars: int) -> List[dict]:
    """Keep newest messages within a rough char budget."""
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
    """Summarize older turns into 'summary' if too many messages. Always uses MINI."""
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
            temperature=0.2,
            max_tokens=450,
        )
        state["summary"] = new_summary.strip()
    except Exception as e:
        print("[CHAT] summarization failed:", e)

    state["messages"] = keep


@app.post("/chat-api")
async def chat_api(
    user_id: str = Form(...),
    message: str = Form(...),
    debug: bool = Form(False),
):
    """
    Stateful chat per user_id + docs RAG.

    - If the user asks about chat history (memory/meta), answer directly from store.
    - Otherwise: docs-first + fallback + sources appended by server.
    - Option B: MODEL_MINI by default, escalate to MODEL_BIG when needed.
    """
    try:
        import re

        user_id = (user_id or "").strip()
        message = (message or "").strip()

        if not user_id:
            return JSONResponse({"error": "Missing user_id"}, status_code=400)
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        # ---- Access control ----
        access_folders = ["documents/public"]
        if user_id.lower().endswith("@erp-is.com"):
            access_folders.append("documents/internal")
        elif INTERNAL_USERS.get(user_id):
            access_folders.append("documents/internal")

        # ---- Conversation state ----
        state = _get_user_state(user_id)

        # Append user message first
        state["messages"].append({"role": "user", "content": message, "ts": _now_iso()})

        # Summarize older turns if needed (MINI)
        _summarize_if_needed(user_id)

        # -----------------------------
        # 1) CHAT MEMORY / META QUERIES
        # -----------------------------
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

        # ----------------------------------
        # 2) NORMAL FLOW: DOCS RAG + MEMORY
        # ----------------------------------
        summary = (state["summary"] or "").strip()
        recent = _trim_messages_to_budget(state["messages"], budget_chars=CHAT_CONTEXT_CHAR_BUDGET)

        # Retrieve chunks WITH sources
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
            "\"The documentation does not mention this, but here is what I know from general knowledge:\"\n\n"
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

        # Option B routing
        fast_force = should_escalate_fast(message, has_docs=has_docs)
        triage = triage_route(
            user_text=message,
            mode="web_chat",
            has_docs=has_docs,
            context_hint=(f"Summary:\n{summary}\n\nRecent turns count: {len(recent)}") if summary else f"Recent turns count: {len(recent)}",
        )

        use_mini = (
            (not fast_force)
            and triage.get("route") == "mini"
            and float(triage.get("confidence", 0.0)) >= TRIAGE_CONFIDENCE_THRESHOLD
        )
        model = MODEL_MINI if use_mini else MODEL_BIG

        answer = chat_completion(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=900,
        )

        # Append sources ONLY if docs-based (no fallback marker)
        fallback_marker = "the documentation does not mention this, but here is what i know from general knowledge:"
        if answer.lower().startswith(fallback_marker):
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

        payload = {"answer": final_answer, "user_id": user_id, "sources": out_sources}
        if debug:
            payload["routed_model"] = model
            payload["triage"] = triage
            payload["fast_force_escalate"] = fast_force
            payload["threshold"] = TRIAGE_CONFIDENCE_THRESHOLD
            payload["has_docs"] = has_docs
            payload["models"] = {"mini": MODEL_MINI, "big": MODEL_BIG}

        return payload

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
    user_email = "default@erp-is.com"  # internal identity for Slack

    try:
        # one-shot (/ask) now auto-routes mini/big
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
