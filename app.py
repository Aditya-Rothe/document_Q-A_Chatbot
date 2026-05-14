import streamlit as st
import tempfile
import os

from rag_pipeline import prepare_vectorstore, ask_question

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Document Q&A",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #6c63ff;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 8px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown("# 🧠 DocMind")
st.caption("Chat with any PDF — powered by LangChain · HuggingFace · FAISS · Gemma 3")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document.png", width=64)
    st.markdown("## 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Supported format: PDF",
        type="pdf",
        help="Upload any PDF — research paper, report, contract, resume etc."
    )

    st.divider()

    st.markdown("### ⚙️ How it works")
    st.markdown("""
1. 📄 **Upload** your PDF
2. ✂️ **Chunked** into 500-token pieces
3. 🔢 **Embedded** with MiniLM-L6-v2
4. 🔍 **Retrieved** via FAISS similarity search
5. 🤖 **Answered** by Gemma 3 27B
    """)

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
- `LangChain` — RAG pipeline
- `FAISS` — vector search
- `HuggingFace` — embeddings + LLM
- `Streamlit` — UI
    """)
    st.divider()
    st.caption("Built by Aditya · [GitHub](https://github.com/yourusername/doc-qa-chatbot)")

# ── Process uploaded PDF ─────────────────────────────────────────────
if uploaded_file:
    if "vectorstore" not in st.session_state or \
       st.session_state.get("file_name") != uploaded_file.name:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Reading and indexing your PDF..."):
            vectorstore = prepare_vectorstore(tmp_path)
            st.session_state.vectorstore = vectorstore
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

        os.unlink(tmp_path)
        st.success(f"✅ **{uploaded_file.name}** is ready — ask anything!")

# ── Chat Interface ────────────────────────────────────────────────────
if "vectorstore" in st.session_state:

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask_question(st.session_state.vectorstore, question)

            st.write(answer)

            if docs:
                with st.expander("📎 Source chunks used to answer"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"""
<div class="source-box">
<strong>Chunk {i} — Page {page}</strong><br>{doc.page_content[:300]}...
</div>
""", unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

    # Clear chat
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
<div style='text-align:center; padding: 3rem 1rem; color: #888;'>
    <div style='font-size: 4rem;'>📄</div>
    <h3 style='color:#555;'>No document uploaded yet</h3>
    <p>Upload a PDF from the sidebar to start chatting with it.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 💡 What can you do with DocMind?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**📑 Research Papers**
Upload any arxiv paper and ask questions about methodology, results, or conclusions.
        """)
    with col2:
        st.markdown("""
**📜 Legal Documents**
Query contracts, agreements, or policy documents in plain English.
        """)
    with col3:
        st.markdown("""
**📊 Business Reports**
Extract KPIs, summaries, and insights from lengthy reports instantly.
        """)