# 📑 Terrorism Research Q&A (RAG Powered)

An AI-powered **Retrieval-Augmented Generation (RAG)** application for exploring terrorism-related research papers.  
Built with **Streamlit**, **LangChain**, and **OpenAI**, this app allows users to query large research documents and get **summarized, citation-backed answers** instantly.



## ⚡ Features
- 📂 Upload terrorism-related research PDFs  
- 🔎 Ask natural language questions  
- 📑 Get AI-generated answers with sources & summaries  
- ⚡ Uses RAG pipeline with embeddings + vector store  
- 🐳 Docker-ready for easy deployment  



## 📂 Project Structure

```
terrorism-research-rag/
│── app.py # Streamlit main app
│── Dockerfile # Containerization
│── requirements.txt # Python dependencies
│── data/
│ └── pdf/ # Research PDFs
│── src/
│ ├── loaders.py # PDF loading
│ ├── embeddings.py # Embedding generation
│ ├── retriever.py # Retriever logic
│ ├── vector_store.py # Vector DB manager
│ └── rag_advpipeline.py # Advanced RAG pipeline

```

## 1️⃣ Clone Repository

git clone https://github.com/Shayan03447/Terrorism-Research-Rag
power shell: code .

## 🎯 Example Workflow

- Add your research PDFs in data/pdf/
- Start the app
- Ask a question like:

# Get:

✅ AI-generated answer
✅ Short summary
✅ Cited sources with page numbers

## 🛠️ Tech Stack

- Python 3.10
- Streamlit (UI)
- LangChain + OpenAI (LLM & embeddings)
- Vector Store (custom)
- Docker (deployment)

## 📌 Future Improvements

- Multi-document upload support
- Fine-tuned domain-specific models
- Enhanced visualization of retrieved chunks