import json
import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Loading FAISS index...")

# Load FAISS index
index = faiss.read_index("../embeddings/faiss_index/index.faiss")

print("Loading metadata...")

# Load metadata list
with open("../embeddings/faiss_index/metadata.json", "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

print("Loading embedding model...")

# Free local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("RAG pipeline ready!")


# ---------------------------------------
# Convert text to embedding vector
# ---------------------------------------
def get_embedding(text):
    vector = embedding_model.encode(text).astype("float32")
    return vector


# ---------------------------------------
# Search FAISS for similar chunks
# ---------------------------------------
def search_similar_chunks(query, top_k=3):
    query_vector = get_embedding(query)
    query_vector = np.expand_dims(query_vector, axis=0)

    distances, indexes = index.search(query_vector, top_k)

    results = []
    for i in indexes[0]:
        results.append(metadata_list[i])
    return results


# ---------------------------------------
# Ask Gemini using retrieved context
# ---------------------------------------
def ask_gemini(question):
    chunks = search_similar_chunks(question, top_k=3)

    # Build context from retrieved chunks
    context_text = ""
    for c in chunks:
        context_text += c["chunk"] + "\n\n"

    # Prompt for the LLM
    prompt = f"""
You are a helpful medical assistant.
Answer the question ONLY using the information from the context below.

Context:
{context_text}

Question: {question}

Give a clear medical answer.
Then list the specialty and sample name of each chunk used.
"""

    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)

    return response.text, chunks

if __name__ == "__main__":
    print("RAG test mode:")
    q = input("Ask a medical question: ")

    ans, src = ask_gemini(q)

    print("\n--- Answer ---")
    print(ans)

    print("\n--- Sources Used ---")
    for s in src:
        print(f"{s['specialty']} | {s['sample_name']} | chunk {s['chunk_number']}")
