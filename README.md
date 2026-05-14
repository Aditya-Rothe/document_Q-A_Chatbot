# 🧠 DocMind — AI PDF Q&A Chatbot

DocMind is an AI-powered RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and ask questions in natural language.

Built using LangChain, FAISS, Sentence Transformers, Groq LLM API, and Streamlit.

Click on this link to use the app- https://documentq-achatbot-tvevnznmz6mwgfbtystgmk.streamlit.app/

---

## 🚀 Features

- 📄 Upload any PDF document
- 💬 Ask questions in natural language
- 🔍 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Source chunk retrieval for transparency
- ⚡ Fast inference using Groq API
- 🎨 Modern Streamlit chat UI

---

## 🛠️ Tech Stack

- **Python 3.12**
- **LangChain**
- **FAISS Vector Database**
- **Sentence Transformers**
- **Groq API**
- **Streamlit**
- **PyPDF**
- **HuggingFace Embeddings**

---

## 📂 Project Structure

```bash
DocMind/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Aditya-Rothe/document_Q-A_Chatbot.git

cd document_Q-A_Chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload PDF document
2. PDF is split into smaller text chunks
3. Chunks are converted into embeddings
4. FAISS performs semantic similarity search
5. Relevant chunks are sent to Groq LLM
6. AI generates contextual answer

---

## 📸 Demo

### Upload PDF
- Upload research papers, reports, resumes, contracts, etc.

### Ask Questions
Examples:
- “Summarize this paper”
- “What are the key findings?”
- “What does the conclusion say?”

---

## 🔥 Example Use Cases

- 📑 Research Paper Assistant
- 📜 Legal Document Analysis
- 📊 Business Report Summarization
- 🎓 Study Notes Q&A
- 🧾 Resume/Portfolio Analysis

---

## 🌐 Deployment

This app can be deployed easily on:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway

---

## 📌 Future Improvements

- Multi-PDF Chat
- Chat Memory
- OCR Support
- Voice Input
- Authentication
- Persistent Vector Database
- Citation Highlighting

---

## 👨‍💻 Author

Aditya Rothe

GitHub:
https://github.com/Aditya-Rothe

---

## ⭐ If You Like This Project

Give this repository a star ⭐
