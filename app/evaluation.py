import time
import json
import faiss
from sentence_transformers import SentenceTransformer
from rag_pipeline import ask_gemini

output_file = "evaluation_results.txt"
f = open(output_file, "w", encoding="utf-8")

print("Loading FAISS index...")
index = faiss.read_index("../embeddings/faiss_index/index.faiss")

print("Loading metadata...")
with open("../embeddings/faiss_index/metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("RAG pipeline ready!\n")

#  10 Evaluation Questions
questions = [
    "What are common symptoms of pneumonia?",
    "How is appendicitis usually diagnosed?",
    "What symptoms are commonly seen in urinary tract infections?",
    "Describe typical postoperative findings in knee arthroscopy.",
    "What are signs of gastrointestinal bleeding?",
    "What symptoms indicate deep vein thrombosis?",
    "What symptoms are typical for COPD exacerbation?",
    "What are common physical findings in skin infections?",
    "How is congestive heart failure documented in clinical notes?",
    "Describe documentation of migraine symptoms."
]

print("\nRunning Evaluation on 10 Medical Queries...\n")

for q in questions:
    try:
        print(f"🔍 Question: {q}")
        f.write(f"\n\n### Question: {q}\n")

        answer, sources = ask_gemini(q)

        print("Answer:", answer)
        f.write(f"Answer:\n{answer}\n")

        f.write("\nSources Used:\n")
        for s in sources:
            f.write(f"- Specialty: {s['specialty']} | Sample: {s['sample_name']}\n")

        print("***\n")

    except Exception as e:
        print("Error:", e)
        f.write(f"\nERROR: {e}\n")

f.close()
print(f"\n Evaluation Completed. File saved as: {output_file}")
