import streamlit as st
from rag_pipeline import ask_gemini

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")

st.title("Medical RAG Q/A System")
st.write("Ask any medical question. The answer will be generated using your medical dataset.")

user_question = st.text_input("Enter your question")

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please type something.")
    else:
        st.write("⏳ Searching medical notes...")

        answer, sources = ask_gemini(user_question)

        st.subheader("💬 Answer")
        st.write(answer)

        st.subheader("📚 Sources Used")
        for s in sources:
            st.write(f"**Specialty:** {s['specialty']}  |  **Sample:** {s['sample_name']}  |  **Chunk:** {s['chunk_number']}")
            st.write(s["chunk"])
            st.write("---")
