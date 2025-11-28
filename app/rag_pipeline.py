import os
import json
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

base_path = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(base_path, "..", "embeddings", "faiss_index", "index.faiss")
meta_path = os.path.join(base_path, "..", "embeddings", "faiss_index", "metadata.json")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

faiss_index = None
meta_data = None
embed_model = None

def load_store():
    global faiss_index, meta_data

    if faiss_index is None:
        faiss_index = faiss.read_index(faiss_path)

    if meta_data is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

def load_embedder():
    global embed_model

    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def make_embed(text):
    load_embedder()
    return embed_model.encode(text).astype("float32")

def find_chunks(text, k=3):
    load_store()

    q_vec = make_embed(text)
    q_vec = np.expand_dims(q_vec, axis=0)

    dist, idx = faiss_index.search(q_vec, k)

    out = []
    for i in idx[0]:
        if i < len(meta_data):
            out.append(meta_data[i])

    return out


def ask_gemini(question):
    chunks = find_chunks(question, k=3)

    ctx = ""
    for c in chunks:
        ctx += c["chunk"] + "\n\n"

    prompt = f"""
Only answer using the context below.

Context:
{ctx}

Question: {question}

Give:
1. Simple medical answer.
2. Specialty + sample name for each chunk.
"""

    model = genai.GenerativeModel("models/gemini-flash-latest")
    reply = model.generate_content(prompt)

    return reply.text, chunks


if __name__ == "__main__":
    print("RAG Test:")
    q = input("Ask something: ")
    ans, src = ask_gemini(q)

    print("\nAnswer:\n", ans)
    print("\nSources:")
    for s in src:
        print(s["specialty"], "|", s["sample_name"], "| chunk", s["chunk_number"])
