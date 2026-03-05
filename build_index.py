import os
import time
import gc
import sqlite3
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pdfplumber
import docx
from tqdm import tqdm
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

CHUNK_SIZE_CHARS = 1000
MAX_CHARS_PER_CHUNK = 6000
BATCH_SIZE = 16
MAX_PAGES = 150

# Fail fast if missing
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_text_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        n = min(len(pdf.pages), MAX_PAGES)
        for i in range(n):
            page = pdf.pages[i]
            t = page.extract_text()
            if t:
                parts.append(t)

            # periodic GC while iterating pages
            if i > 0 and i % 25 == 0:
                gc.collect()

    return "\n".join(parts)


def extract_text_docx(path: Path) -> str:
    d = docx.Document(str(path))
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    if ext == ".docx":
        return extract_text_docx(path)
    return ""


def chunk_text(text: str, max_chars: int = CHUNK_SIZE_CHARS) -> List[str]:
    """
    Sentence-ish splitter with hard fallback for long segments.
    Works better on bullet lists / tables than split(". ").
    """
    text = (text or "").strip()
    if not text:
        return []

    # Split on newlines first (often better than ". " for docs)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for ln in lines:
        # If a single line is extremely long, hard-split it
        if len(ln) > max_chars:
            # flush current
            flush()
            for i in range(0, len(ln), max_chars):
                part = ln[i:i + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        candidate = (buf + "\n" + ln).strip() if buf else ln
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush()
            buf = ln

    flush()
    return chunks


def clean_chunks(chunks: List[str]) -> List[str]:
    clean: List[str] = []
    for ch in chunks:
        if not ch:
            continue
        ch = ch.strip()
        if not ch:
            continue
        if len(ch) > MAX_CHARS_PER_CHUNK:
            ch = ch[:MAX_CHARS_PER_CHUNK]
        # remove super tiny chunks
        if len(ch) < 20:
            continue
        clean.append(ch)
    return clean


def embed_batch(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = min(2 ** attempt, 20)
            print(f"[EMBED] error (attempt {attempt+1}/{max_retries}): {e} -> sleep {wait}s", flush=True)
            time.sleep(wait)

    # give up this batch
    return []


def build_index(docs_path: str, out_path: str):
    docs_path = Path(docs_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # SQLite
    db_path = out_path / "meta.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT NOT NULL,
            chunk TEXT NOT NULL
        )
    """)
    conn.commit()

    # FAISS
    index = faiss.IndexFlatL2(VECTOR_DIM)
    faiss_path_tmp = out_path / "faiss.index.tmp"
    faiss_path_final = out_path / "faiss.index"

    files = sorted(list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.docx")))
    print(f"Found {len(files)} files (PDF+DOCX) under {docs_path}", flush=True)

    inserted = 0
    pending_rows: List[Tuple[str, str]] = []

    for f in tqdm(files):
        try:
            print(f"Processing file: {f}", flush=True)

            text = extract_text(f)
            if not text.strip():
                continue

            chunks = clean_chunks(chunk_text(text))
            if not chunks:
                continue

            f_str = str(f)

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                embs = embed_batch(batch_chunks)

                if not embs:
                    print(f"[WARN] embeddings failed for batch in {f} (skipping batch)", flush=True)
                    continue

                if len(embs) != len(batch_chunks):
                    print(f"[WARN] embedding count mismatch in {f}: got {len(embs)} for {len(batch_chunks)} (skipping batch)", flush=True)
                    continue

                vecs = np.array(embs, dtype="float32")
                index.add(vecs)

                for ch in batch_chunks:
                    pending_rows.append((f_str, ch))

                inserted += len(batch_chunks)

                if len(pending_rows) >= 2000:
                    cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", pending_rows)
                    conn.commit()
                    pending_rows.clear()

            # crash-safety: write tmp after each file
            faiss.write_index(index, str(faiss_path_tmp))

        except Exception as e:
            print(f"[ERROR] failed processing {f}: {e}", flush=True)

        finally:
            # Aggressive cleanup between files
            gc.collect()

    if pending_rows:
        cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", pending_rows)
        conn.commit()
        pending_rows.clear()

    faiss.write_index(index, str(faiss_path_final))
    if faiss_path_tmp.exists():
        try:
            faiss_path_tmp.unlink()
        except Exception:
            pass

    conn.close()

    print(f"✅ Index build complete: {faiss_path_final} + {db_path}", flush=True)
    print(f"Total vectors: {index.ntotal}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_index(args.docs, args.out)
