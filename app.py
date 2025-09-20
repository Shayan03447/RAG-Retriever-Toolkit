import streamlit as st
from src.loaders import load_pdf, split_documents
from src.embeddings import EmbeddingManager
from src.rag_advpipeline import AdvancedRAGPipeline
from src.retriever import RagRetriever
from src.vector_store import VectorStore
from langchain_openai import ChatOpenAI


# -----Setup--- Add title and Branding
st.set_page_config(page_title="Terrorism Research Q&A", page_icon="📑")
st.title("📑 Terrorism Research Q&A")
st.write("Ask question from terrorism-related researched paper using RAG-powered retrievel and summarization.")

# ----- Load_pdf and prepare data 

@st.cache_resource
def setup_pipeline():

    # Load: PDF'S
    pdf_file=load_pdf("data/pdf")
    chunks=split_documents(pdf_file)

    # Embeddings
    embedding_manager=EmbeddingManager()
    texts=[doc.page_content for doc in chunks]
    embeddings=embedding_manager.generate_embeddings(texts)

    # Vector-Store
    vector_store=VectorStore()
    vector_store.add_documents(chunks, embeddings)

    # Retriever
    retriever=RagRetriever(vector_store, embedding_manager)

    # Advanced Pipeline with LLM
    rag_pipeline=AdvancedRAGPipeline(retriever)

    return rag_pipeline

rag=setup_pipeline()

# ----User input(Query-Box)

question=st.text_input("🔎Enter your research question:")

# -------Run Query
if st.button("Get Answer"):
    if question:
        result=rag.query(question, top_k=3, min_score=0.3, summarize=True,)
        # Show Answer
        st.subheader("Answer")
        st.write(result['answer'])

        # Show summary
        if result["summary"]:
            st.subheader("Summary")
            st.write(result["summary"])
        
        # Show Sources
        st.subheader("Sources")
        for src in result["sources"]:
            st.markdown(f"- **{src['source']}**, Page {src['page']} (Score: {src['score']:.2f})")
            st.caption(src["preview"])
    else:
        st.warning("Please enter a Question")



