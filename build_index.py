import os
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_text_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip()
    except Exception as e:
        print(f"[WARN] failed to read PDF {path}: {e}")
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
        batch = clean_texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            vectors.extend([d.embedding for d in resp.data])
        except Exception as e:
            print(f"[WARN] embedding batch failed: {e}")

    return vectors


def build_rows_from_pdfs(docs_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []

    files = sorted(docs_path.rglob("*.pdf"))
    print(f"Found {len(files)} PDF file(s) under {docs_path}")

    for file_path in files:
        print(f"Reading: {file_path}")
        text = extract_text_pdf(file_path)
        if not text:
            continue

        chunks = chunk_text(text, max_chars=1000)
        for chunk in chunks:
            if chunk.strip():
                rows.append((str(file_path), chunk.strip()))

    return rows


def write_meta_db(db_path: Path, rows: List[Tuple[str, str]]):
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT NOT NULL,
                chunk TEXT NOT NULL
            )
            """
        )
        cur.executemany(
            "INSERT INTO chunks (file, chunk) VALUES (?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def build_index(docs_dir: str, out_dir: str):
    docs_path = Path(docs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs folder does not exist: {docs_path}")

    rows = build_rows_from_pdfs(docs_path)
    if not rows:
        raise RuntimeError(f"No text chunks found in {docs_path}")

    embed_inputs = [
        f"Source file: {os.path.basename(file_path)}\n\nContent:\n{chunk}"
        for file_path, chunk in rows
    ]

    print(f"Embedding {len(embed_inputs)} chunk(s)...")
    vectors = embed_texts(embed_inputs)
    if not vectors:
        raise RuntimeError("No embeddings were generated")

    if len(vectors) != len(rows):
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} embeddings for {len(rows)} rows"
        )

    vecs = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vecs)

    faiss_path = out_path / "faiss.index"
    db_path = out_path / "meta.db"

    print(f"Writing FAISS index to {faiss_path}")
    faiss.write_index(index, str(faiss_path))

    print(f"Writing sqlite metadata to {db_path}")
    write_meta_db(db_path, rows)

    stats = {
        "docs_dir": str(docs_path),
        "out_dir": str(out_path),
        "files_indexed": len(sorted(docs_path.rglob('*.pdf'))),
        "chunks_indexed": len(rows),
        "embedding_model": EMBEDDING_MODEL,
        "vector_dim": VECTOR_DIM,
        "faiss_metric": "IndexFlatIP with L2-normalized vectors (cosine-style retrieval)",
    }

    with open(out_path / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Index build complete")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Source docs directory, e.g. /data/documents/public")
    parser.add_argument("--out", required=True, help="Output index directory, e.g. /data/indexes/public")
    args = parser.parse_args()

    build_index(args.docs, args.out)


if __name__ == "__main__":
    main()
