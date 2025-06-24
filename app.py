import os
import json
import streamlit as st
import PyPDF2
import hashlib
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# === CONFIG ===
DATA_DIR = "dataset"
FAISS_DIR = "faiss_store"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# === LOAD MODELS ===
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

embed_model = load_model()

# === LOAD FAISS INDEX ===
@st.cache_resource
def load_faiss():
    index = faiss.read_index(os.path.join(FAISS_DIR, "decidim.index"))
    with open(os.path.join(FAISS_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_faiss()

# === RETRIEVE ===
def search(question, top_k=3):
    query_vec = embed_model.encode([question])
    distances, indices = index.search(query_vec, top_k)
    results = [metadata[i] for i in indices[0]]
    context = "\n\n".join([f"{r['text']} (📄 {r['source']})" for r in results])
    return context

# === STREAMING TOGETHER ===
def call_together_stream(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": True
    }
    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body, stream=True)

    reply = ""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8").replace("data: ", "")
            try:
                data = json.loads(line)
                delta = data["choices"][0]["delta"]
                content = delta.get("content", "")
                reply += content
                yield content
            except:
                continue

# === RAG PIPELINE ===
def answer_question(question):
    context = search(question)
    prompt = f"""First understand the question itseld, wether its a general question about your introduction or they want to know something about Decidim. If user greet you, reply as an helpful assistent who will help them to understand the Decidim tool through interaction with you. You are an assistant that answers only based on the provided context from Decidim documentation.
                    Do not guess or add any information not present in the context. If the answer is not in the context,
                    respond with: "This is beyond my current knowledge base."

Context:
{context}

Question: {question}
Answer:"""
    return call_together_stream(prompt), context

# === STREAMLIT UI ===
st.set_page_config(page_title="Decidim Chatbot", layout="centered")
st.title("🗳️ Decidim Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask about Decidim...")
if query:
    st.session_state.chat.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        gen, context = answer_question(query)
        full_response = ""
        response_box = st.empty()
        for chunk in gen:
            full_response += chunk
            response_box.markdown(full_response)
        st.expander("📚 Source Context").markdown(context)
        st.session_state.chat.append(("assistant", full_response))
