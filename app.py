import streamlit as st
import tempfile
import os

# Updated imports to match the new rag_pipeline structure
from rag_pipeline import (
    prepare_vectorstore,
    ask_question
)

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Chatbot")
st.caption("LangChain + Hugging Face + FAISS")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type="pdf"
    )

# ------------------------------------------------
# Process PDF
# ------------------------------------------------
if uploaded_file:
    # Check if a new file is uploaded
    if "vectorstore" not in st.session_state or \
       st.session_state.get("file_name") != uploaded_file.name:

        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Processing PDF..."):
            # Use the new combined function from rag_pipeline
            # This handles loading, splitting, and building the index
            vectorstore = prepare_vectorstore(tmp_path)

            # Store the vectorstore directly in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

        # Clean up the temp file
        os.unlink(tmp_path)
        st.success("✅ PDF processed successfully!")

# ------------------------------------------------
# Chat UI
# ------------------------------------------------
if "vectorstore" in st.session_state:
    # Display message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask question about PDF...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call ask_question using the vectorstore
                answer, docs = ask_question(
                    st.session_state.vectorstore,
                    question
                )

                if answer:
                    st.write(answer)
                    
                    with st.expander("Source Chunks"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"### Chunk {i}")
                            st.write(doc.page_content)
                            st.markdown("---")
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    st.error("Failed to get a response from the model.")

else:
    st.info("Upload a PDF to begin.")