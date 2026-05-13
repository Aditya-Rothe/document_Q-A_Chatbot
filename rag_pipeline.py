import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load .env
load_dotenv()


# ---------------------------------------------------
# Get Hugging Face Token
# ---------------------------------------------------
def get_hf_token():

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        st.error("❌ Hugging Face token not found in .env")
        st.stop()

    return token


# ---------------------------------------------------
# Create Vector Store
# ---------------------------------------------------
@st.cache_resource
def prepare_vectorstore(pdf_path: str):

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS Vector Store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------
def ask_question(vectorstore, question: str):

    token = get_hf_token()

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    # Combine chunks into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Use ONLY the provided context to answer. "
                "If the answer is not in the context, say: "
                "'I don't have enough information in the document to answer this.'"
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]

    try:

        # Hugging Face Client
        client = InferenceClient(
            token=token
        )

        # Generate Response
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )

        answer = completion.choices[0].message.content

        return answer, docs

    except Exception as e:

        st.error(f"Inference Error: {e}")

        return (
            "The AI server is currently busy. Please try again.",
            []
        )