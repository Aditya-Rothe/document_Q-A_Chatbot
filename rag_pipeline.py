import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def get_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except Exception:
            pass
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    else:
        st.error("❌ HuggingFace token not found.")
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

    # Using HuggingFace InferenceClient with nebius provider
    # — confirmed working free tier as of 2025
    client = InferenceClient(
        provider="nebius",
        api_key=token,
    )

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip(), docs

    except Exception as e1:
        # Fallback — smaller model, same provider
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip(), docs

        except Exception as e2:
            st.error(f"Both models failed.\nPrimary: {e1}\nFallback: {e2}")
            return "The AI server is currently busy. Please try again in a moment.", []