from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def load_pdf(data_path: str = "/data/pdf"):
    """
    What: Load all the pdf from the given folder
    why: RAG Pipeline knowladge base
    """
    dir_loader=DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )
    documents=dir_loader.load()
    print(f"[Info] loaded {len(documents)} pages from pdf(s)")
    return documents

def split_documents(pdf_documents, chunk_size=1000, chunk_overlap=200):
    "Split documents into smaller chunks for better rag performance"
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )
    chunks=text_splitter.split_documents(pdf_documents)
    print(f"[INFO] Split {len(pdf_documents)} pages into {len(chunks)} text chunks")
    return chunks
    