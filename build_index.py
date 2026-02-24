import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
import docx
from tqdm import tqdm
from openai import OpenAI
import gc

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

CHUNK_SIZE_CHARS = 1000
MAX_CHARS_PER_CHUNK = 6000
BATCH_SIZE = 16
MAX_PAGES = 150

client = OpenAI()




def extract_text_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for i in range(min(len(pdf.pages), MAX_PAGES)):
            page = pdf.pages[i]
            t = page.extract_text()
            if t:
                parts.append(t)
            # mini garde-fou : flush toutes les 25 pages
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


def chunk_text(text: str, max_chars: int = CHUNK_SIZE_CHARS) -> list[str]:
    sentences = text.split(". ")
    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        candidate = (current + s + ". ").strip()
        if len(candidate) <= max_chars:
            current = candidate + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = (s + ". ").strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def clean_chunks(chunks: list[str]) -> list[str]:
    clean = []
    for ch in chunks:
        if not ch:
            continue
        ch = ch.strip()
        if not ch:
            continue
        if len(ch) > MAX_CHARS_PER_CHUNK:
            ch = ch[:MAX_CHARS_PER_CHUNK]
        clean.append(ch)
    return clean


def embed_batch(texts: list[str]) -> list[list[float]]:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


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

    # FAISS (streaming)
    index = faiss.IndexFlatL2(VECTOR_DIM)
    faiss_path_tmp = out_path / "faiss.index.tmp"
    faiss_path_final = out_path / "faiss.index"

    files = sorted(list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.docx")))
    print(f"Found {len(files)} files (PDF+DOCX) under {docs_path}")

    inserted = 0
    pending_rows: list[tuple[str, str]] = []

    for f in tqdm(files):
        print(f"Processing file: {f}", flush=True)
        text = extract_text(f)
        if not text.strip():
            continue

        chunks = clean_chunks(chunk_text(text))
        if not chunks:
            continue

        # embed & add in FAISS by micro-batch
        f_str = str(f)
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            embs = embed_batch(batch_chunks)

            if len(embs) != len(batch_chunks):
                raise RuntimeError(f"Embedding count mismatch for {f}: got {len(embs)} for {len(batch_chunks)}")

            # Add to FAISS immediately (low RAM)
            vecs = np.array(embs, dtype="float32")
            index.add(vecs)

            # Add to SQLite buffer
            for ch in batch_chunks:
                pending_rows.append((f_str, ch))

            inserted += len(batch_chunks)

            # flush sqlite regularly
            if len(pending_rows) >= 2000:
                cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", pending_rows)
                conn.commit()
                pending_rows.clear()

        # optional: write partial index occasionally (crash safety)
        if inserted % 5000 == 0:
            faiss.write_index(index, str(faiss_path_tmp))

    # flush rest
    if pending_rows:
        cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", pending_rows)
        conn.commit()
        pending_rows.clear()

    # final write
    faiss.write_index(index, str(faiss_path_final))
    if faiss_path_tmp.exists():
        faiss_path_tmp.unlink(missing_ok=True)

    conn.close()

    print(f"✅ Index build complete: {faiss_path_final} + {db_path}")
    print(f"Total vectors: {index.ntotal}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_index(args.docs, args.out)
