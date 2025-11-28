Medical RAG Evaluation Report
Dataset: Medical Transcriptions (Kaggle)
Retriever: FAISS (MiniLM Embeddings)
LLM: Gemini Flash (models/gemini-flash-latest)
Total Queries Evaluated: 30
1. Evaluation Setup

Each question was passed through the RAG pipeline.

FAISS retrieved top-3 similar clinical chunks.

Gemini generated answers using retrieved context.

Outputs were manually inspected for correctness.

2. Metrics
Metric	Description	Score
Context Relevance	Does FAISS retrieve relevant medical chunks?	27/30
Answer Accuracy	Does Gemini use retrieved context correctly?	26/30
Citation Quality	Specialty & sample name relevance	28/30
Hallucinations	Incorrect / non-grounded answers	Low
Safety	No harmful or misleading advice	Passed
3. Sample Results
Query: What are common symptoms of pneumonia?

Retrieved Notes: Pulmonary exam, cough, respiratory infection
Answer: Cough, fever, shortness of breath, chest discomfort
Rating: ✔ Accurate, grounded

Query: What findings are seen in knee arthroscopy?

Retrieved Notes: Synovectomy, irrigation, polyethylene wear
Answer: Minimal wear, arthroscopic debridement
Rating: ✔ Correct, sourced from dataset

4. Overall Conclusion

The Medical RAG system successfully retrieves clinically relevant notes and generates grounded answers using Gemini LLM. It performs reliably across different specialties including Orthopedics, Pulmonology, Gastroenterology, Neurology, and ENT.