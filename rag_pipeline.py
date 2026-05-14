import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------
# Load Environment Variables
# ------------------------------------------------
load_dotenv()


# ------------------------------------------------
# Get Groq API Key
# ------------------------------------------------
def get_groq_api_key():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.stop()

    return api_key


# ------------------------------------------------
# Prepare Vector Store
# ------------------------------------------------
@st.cache_resource
def prepare_vectorstore(pdf_path: str):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ------------------------------------------------
# Ask Question
# ------------------------------------------------
def ask_question(vectorstore, question: str):

    try:

        # Get API key
        api_key = get_groq_api_key()

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # Retrieve relevant docs
        docs = retriever.invoke(question)

        # Build context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Prompt
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the document to answer this."

Context:
{context}

Question:
{question}

Answer:
"""

        # Initialize Groq client
        client = Groq(api_key=api_key)

        # Generate response
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
        )

        answer = chat_completion.choices[0].message.content

        return answer, docs

    except Exception as e:

        st.error(f"Error: {e}")

        return (
            "❌ Failed to generate answer.",
            []
        )