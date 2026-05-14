import streamlit as st
import tempfile
import os

from rag_pipeline import prepare_vectorstore, ask_question

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — AI PDF Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }

    .stChatMessage {
        border-radius: 14px;
        padding: 8px;
    }

    .source-box {
        background: #f8f9fa;
        border-left: 4px solid #6c63ff;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 10px;
        color: #333;
    }

    .hero-box {
        text-align: center;
        padding: 3rem 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #f8f9ff, #eef1ff);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #222;
    }

    .hero-subtitle {
        color: #666;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🧠 DocMind</div>
    <div class="hero-subtitle">
        Chat with your PDFs using AI-powered Retrieval-Augmented Generation (RAG)
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Research papers, reports, resumes, legal docs, notes, etc."
    )

    st.divider()

    st.markdown("### ⚙️ How It Works")

    st.markdown("""
1. 📄 Upload PDF  
2. ✂️ Split into chunks  
3. 🔢 Convert text into embeddings  
4. 🔍 Retrieve relevant context using FAISS  
5. 🤖 Generate answer using Groq LLM  
    """)

    st.divider()

    st.markdown("### 🛠️ Tech Stack")

    st.markdown("""
- LangChain  
- FAISS Vector Store  
- Sentence Transformers  
- Groq API  
- Streamlit  
    """)

    st.divider()

    st.markdown(
        "Made with ❤️ by Aditya"
    )

# ── Process PDF ──────────────────────────────────────────────────────
if uploaded_file:

    if (
        "vectorstore" not in st.session_state
        or st.session_state.get("file_name") != uploaded_file.name
    ):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Processing PDF and building vector index..."):

            vectorstore = prepare_vectorstore(tmp_path)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

        os.unlink(tmp_path)

        st.success(f"✅ {uploaded_file.name} processed successfully!")

# ── Chat Interface ───────────────────────────────────────────────────
if "vectorstore" in st.session_state:

    # Show chat history
    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("Ask anything about your PDF..."):

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer, docs = ask_question(
                    st.session_state.vectorstore,
                    question
                )

            st.write(answer)

            # Source chunks
            if docs:

                with st.expander("📎 Source Chunks"):

                    for i, doc in enumerate(docs, 1):

                        page = doc.metadata.get("page", "?")

                        st.markdown(f"""
<div class="source-box">
<strong>Chunk {i} — Page {page}</strong><br><br>
{doc.page_content[:350]}...
</div>
""", unsafe_allow_html=True)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # Clear history button
    if st.session_state.get("chat_history"):

        if st.button("🗑️ Clear Chat"):

            st.session_state.chat_history = []
            st.rerun()

# ── Empty State ──────────────────────────────────────────────────────
else:

    st.markdown("""
<div style='text-align:center; padding: 2rem; color:#777;'>

# 📄 Upload a PDF to Begin

Ask questions about:
- Research Papers
- Reports
- Resumes
- Legal Documents
- Study Notes

</div>
""", unsafe_allow_html=True)

    st.markdown("## 💡 Example Use Cases")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
### 📑 Research Papers
Summarize findings, methods, and conclusions instantly.
""")

    with col2:
        st.markdown("""
### 📜 Legal Docs
Ask questions about contracts and agreements in plain English.
""")

    with col3:
        st.markdown("""
### 📊 Business Reports
Extract KPIs, insights, and summaries from long reports.
""")