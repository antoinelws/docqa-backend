import os
import sqlite3
import faiss
import numpy as np
import pdfplumber
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()


MAX_CHARS_PER_CHUNK = 6000   # safe
BATCH_SIZE = 64              # safe
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072
CHUNK_SIZE = 1000

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def chunk_text(text, max_chars=CHUNK_SIZE):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def embed_batch(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_index(docs_path, output_path):

    docs_path = Path(docs_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(VECTOR_DIM)

    db_path = output_path / "meta.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT,
            chunk TEXT
        )
    """)
    conn.commit()

    all_vectors = []

    pdf_files = list(docs_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

for pdf in tqdm(pdf_files):
    text = extract_text(pdf)
    chunks = chunk_text(text)

    # Nettoyage: remove empty + trim + truncate
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

    if not clean:
        continue

    print(f"Processing {pdf} with {len(clean)} chunks")

    # Embed par batch
    embeddings = []
    for i in range(0, len(clean), BATCH_SIZE):
        batch = clean[i:i + BATCH_SIZE]
        embeddings.extend(embed_batch(batch))

    # IMPORTANT: on remplace chunks par clean pour matcher embeddings
    chunks = clean

        for chunk, emb in zip(chunks, embeddings):
            c.execute("INSERT INTO chunks (file, chunk) VALUES (?, ?)",
                      (str(pdf), chunk))
            all_vectors.append(emb)

        conn.commit()

    vectors_np = np.array(all_vectors).astype("float32")
    index.add(vectors_np)

    faiss.write_index(index, str(output_path / "faiss.index"))

    conn.close()
    print("Index build complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_index(args.docs, args.out)
