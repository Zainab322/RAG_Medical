import pandas as pd
import json

print("The dataset is loading...")

df = pd.read_csv("../data/mtsamples.csv")

df = df.dropna(subset=["transcription"])
df = df.head(500)

print("Cleaning text and making chunks...")

chunks = []

chunk_size = 700
chunk_overlap = 50

for index, row in df.iterrows():
    text = str(row["transcription"])
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split()) 

    start = 0
    end = chunk_size

    while start < len(text):
        chunk_text = text[start:end]

        chunk_data = {
            "id": str(index),
            "chunk": chunk_text,
            "specialty": row["medical_specialty"],
            "sample_name": row["sample_name"],
            "chunk_number": start
        }

        chunks.append(chunk_data)

        start = start + (chunk_size - chunk_overlap)
        end = start + chunk_size

print("Total chunks:", len(chunks))

print("Saving chunks")

with open("../data/preprocessed_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Completed")
