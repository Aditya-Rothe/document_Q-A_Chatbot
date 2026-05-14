import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def get_hf_token():
    """Read HF token — works locally (.env) and on Streamlit Cloud (secrets)."""

    # Try .env first (local development)
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Try Streamlit secrets (Streamlit Cloud deployment)
    if not token:
        try:
            token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except Exception:
            pass

    # Set it as env var so all libraries can access it
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    else:
        st.error("❌ HuggingFace token not found. Add it to .env or Streamlit secrets.")
        st.stop()

    return token


@st.cache_resource
def prepare_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ask_question(vectorstore, question: str):
    token = get_hf_token()

    # TEMPORARY DEBUG — remove after confirming it works on Streamlit Cloud
    st.info(f"Debug: token length={len(token)}, starts with hf_={token.startswith('hf_')}")

    # Retrieve top 3 relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use ONLY the context below to answer. "
                "If the answer is not in the context, say: "
                "'I don't have enough information in the document to answer this.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    # HuggingFace official router — works on Streamlit Cloud
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip(), docs

    except Exception as e1:
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip(), docs

        except Exception as e2:
            st.error(f"Both models failed.\nPrimary: {e1}\nFallback: {e2}")
            return "The AI server is currently busy. Please try again in a moment.", []