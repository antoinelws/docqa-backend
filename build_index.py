import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
import docx
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# Config
# -------------------------
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072

CHUNK_SIZE_CHARS = 1000          # découpe "logique"
MAX_CHARS_PER_CHUNK = 6000       # hard cap pour éviter input invalide
BATCH_SIZE = 64                  # batch embeddings

# OpenAI client (lit OPENAI_API_KEY depuis l'env automatiquement)
client = OpenAI()


# -------------------------
# Extraction texte
# -------------------------
def extract_text_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
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


# -------------------------
# Chunking + nettoyage
# -------------------------
def chunk_text(text: str, max_chars: int = CHUNK_SIZE_CHARS) -> list[str]:
    # Chunking simple par phrases (suffisant pour démarrer)
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
        # hard cap pour éviter erreurs input
        if len(ch) > MAX_CHARS_PER_CHUNK:
            ch = ch[:MAX_CHARS_PER_CHUNK]
        clean.append(ch)
    return clean


# -------------------------
# Embeddings
# -------------------------
def embed_batch(texts: list[str]) -> list[list[float]]:
    # Filtre final (parano)
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


# -------------------------
# Index build
# -------------------------
def build_index(docs_path: str, out_path: str):
    docs_path = Path(docs_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # FAISS
    index = faiss.IndexFlatL2(VECTOR_DIM)

    # SQLite metadata
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

    # Collect vectors in RAM then add once (ok vu ton volume)
    all_vectors: list[list[float]] = []
    rows_to_insert: list[tuple[str, str]] = []

    # PDFs + DOCX
    files = sorted(list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.docx")))
    print(f"Found {len(files)} files (PDF+DOCX) under {docs_path}")

    for f in tqdm(files):
        text = extract_text(f)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        chunks = clean_chunks(chunks)
        if not chunks:
            continue

        # embed par batch
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embs = embed_batch(batch)
            # En cas de retour inattendu (rare), on garde la cohérence
            if len(embs) != len(batch):
                raise RuntimeError(f"Embedding count mismatch for {f}: got {len(embs)} for batch {len(batch)}")
            embeddings.extend(embs)

        # store meta + vectors
        f_str = str(f)
        for ch, emb in zip(chunks, embeddings):
            rows_to_insert.append((f_str, ch))
            all_vectors.append(emb)

        # flush sqlite toutes les ~2000 chunks pour éviter mémoire/temps
        if len(rows_to_insert) >= 2000:
            cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", rows_to_insert)
            conn.commit()
            rows_to_insert.clear()

    # flush restant
    if rows_to_insert:
        cur.executemany("INSERT INTO chunks (file, chunk) VALUES (?, ?)", rows_to_insert)
        conn.commit()
        rows_to_insert.clear()

    if not all_vectors:
        conn.close()
        raise RuntimeError("No vectors were created. Check extraction/chunking.")

    vectors_np = np.array(all_vectors, dtype="float32")
    index.add(vectors_np)

    faiss_path = out_path / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    conn.close()
    print(f"✅ Index build complete: {faiss_path} + {db_path}")
    print(f"Total vectors: {vectors_np.shape[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Folder containing documents (pdf/docx)")
    parser.add_argument("--out", required=True, help="Output folder for faiss.index + meta.db")
    args = parser.parse_args()

    build_index(args.docs, args.out)
