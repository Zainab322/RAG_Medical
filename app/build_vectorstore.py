import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading preprocessed chunks...")

with open("../data/preprocessed_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of chunks:", len(data))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_list = []
metadata_list = []

print("Making embeddings...")

for item in data:
    text = item["chunk"]

    vector = model.encode(text).astype("float32")

    embeddings_list.append(vector)
    metadata_list.append(item)

embeddings_array = np.array(embeddings_list)

print("Building FAISS index...")

dimension = embeddings_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings_array)

print("Saving FAISS index...")

os.makedirs("../embeddings/faiss_index", exist_ok=True)

faiss.write_index(index, "../embeddings/faiss_index/index.faiss")

with open("../embeddings/faiss_index/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2)

print("Done!")
