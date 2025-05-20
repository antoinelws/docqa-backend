from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pdfplumber, docx, os, json
import openai
import faiss
import numpy as np

app = FastAPI()

# Allow frontend/mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paste your OpenAI API key here
openai.api_key = "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"

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM = 3072
DOC_FOLDER = "documents"

# Ensure documents folder exists
if not os.path.exists(DOC_FOLDER):
    os.makedirs(DOC_FOLDER)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Use the uploaded filename (without extension) as document name
        filename = file.filename
        document_name = os.path.splitext(filename)[0]

        content = await file.read()
        text = extract_text(filename, content)
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)

        # Save chunks
        with open(os.path.join(DOC_FOLDER, f"{document_name}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        # Save FAISS index
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(np.array(vectors).astype("float32"))
        faiss.write_index(index, os.path.join(DOC_FOLDER, f"{document_name}.index"))

        return {"message": f"Document '{document_name}' saved with {len(chunks)} chunks."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        # Embed the question
        question_vec = embed_texts([question])[0]
        combined_chunks = []

        # Search through every document in folder
        for file in os.listdir(DOC_FOLDER):
            if file.endswith(".index"):
                doc_name = file.replace(".index", "")
                index_path = os.path.join(DOC_FOLDER, file)
                json_path = os.path.join(DOC_FOLDER, f"{doc_name}.json")

                if not os.path.exists(json_path):
                    continue

                try:
                    index = faiss.read_index(index_path)

                    with open(json_path, "r", encoding="utf-8") as f:
                        chunks = json.load(f)

                    D, I = index.search(np.array([question_vec]).astype("float32"), k=3)
                    for score, idx in zip(D[0], I[0]):
                        if 0 <= idx < len(chunks):
                            combined_chunks.append((score, chunks[idx]))

                except Exception as e:
                    print(f"Error reading {doc_name}: {e}")

        # Sort by best scores
        combined_chunks.sort(key=lambda x: x[0])
        top_chunks = [chunk for _, chunk in combined_chunks[:5]]

        if not top_chunks:
            return {"answer": "No relevant content found in any document."}

        prompt = (
            "You are an AI assistant. Use the following document excerpts to answer the question.\n\n"
            + "\n\n".join(top_chunks)
            + f"\n\nQuestion: {question}\nAnswer:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You answer questions based on documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


# ---------- Utility Functions ----------

def extract_text(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(content)
        with pdfplumber.open("temp.pdf") as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == "docx":
        with open("temp.docx", "wb") as f:
            f.write(content)
        doc = docx.Document("temp.docx")
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return "Unsupported file format. Only PDF and DOCX are supported."

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chars:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai.Embedding.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [d["embedding"] for d in response["data"]]
