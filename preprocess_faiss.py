# preprocess_faiss.py
import os, pickle, hashlib
import faiss
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "dataset"
FAISS_DIR = "faiss_store"
os.makedirs(FAISS_DIR, exist_ok=True)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
texts, metadatas = [], []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        with open(os.path.join(DATA_DIR, filename), "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text.strip())
                    metadatas.append({
                        "text": text.strip(),
                        "source": f"{filename} - Page {i+1}"
                    })

embeddings = model.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, os.path.join(FAISS_DIR, "decidim.index"))
with open(os.path.join(FAISS_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadatas, f)

print(f"✅ FAISS index created with {len(texts)} chunks.")
