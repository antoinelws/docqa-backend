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

MODEL_BIG = os.getenv("OPENAI_MODEL_BIG", "gpt-5.2")

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
# OpenAI helper
# =========================
def chat_completion(model: str, messages: List[dict], max_completion_tokens: int = 700) -> str:
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
        raise RuntimeError(f"Authentication failed: {result}")
    return result["access_token"]


# =========================
# Text extraction
# =========================
def extract_text(filename: str, content: bytes) -> str:
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
        return content.decode("utf-8", errors="ignore")

    return ""


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    sentences = text.split(". ")
    chunks, buf = [], ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        candidate = buf + s + ". "
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = s + ". "

    if buf.strip():
        chunks.append(buf.strip())

    return chunks


# =========================
# Embeddings
# =========================
def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    import openai
    openai.api_key = OPENAI_API_KEY

    clean = [t.strip()[:6000] for t in texts if isinstance(t, str) and len(t.strip()) > 10]
    if not clean:
        return []

    vectors = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i:i + batch_size]
        try:
            resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            vectors.extend(d["embedding"] for d in resp["data"])
        except Exception as e:
            print("[EMBED ERROR]", e)

    return vectors


# =========================
# Index cache
# =========================
@lru_cache(maxsize=8)
def load_indexes(folder: str):
    results = []
    for json_path in glob.glob(f"{folder}/*.json"):
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
    load_indexes.cache_clear()


# =========================
# FastAPI
# =========================
app = FastAPI()
app.include_router(estimator_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# One-shot QA
# =========================
@app.post("/ask")
def ask_question(question: str = Form(...), user_email: str = Form(...)):
    access_folders = ["documents/public"]
    if user_email.endswith("@erp-is.com"):
        access_folders.append("documents/internal")

    chunks = []
    sources = []

    qvecs = embed_texts([question])
    if qvecs:
        qvec = qvecs[0]
        for folder in access_folders:
            for base, cks, index in load_indexes(folder):
                D, I = index.search(np.array([qvec]).astype("float32"), k=6)
                for idx in I[0]:
                    if 0 <= idx < len(cks):
                        chunks.append(cks[idx])
                        sources.append(base)

    has_docs = bool(chunks)
    docs_block = "\n---\n".join(chunks)

    system = (
        "You are the ShipERP assistant.\n"
        "Answer strictly from the documentation excerpts below if they contain the answer.\n\n"
        "If not, say exactly:\n"
        "\"The documentation does not mention this, but here is what I know from general knowledge:\""
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Documentation excerpts:\n{docs_block}"},
        {"role": "user", "content": question},
    ]

    answer = chat_completion(MODEL_BIG, messages)

    if answer.lower().startswith("the documentation does not mention"):
        return {"answer": answer, "sources": []}

    return {
        "answer": answer,
        "sources": sorted(set(sources)) if has_docs else []
    }


# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup_event():
    print("ShipERP AI Assistant started")
