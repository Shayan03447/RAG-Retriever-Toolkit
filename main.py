from src.loaders import load_pdf, split_documents
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RagRetriever
from src.rag_pipeline import RagPipeline
# Step 1 load pdf's
pdf_file=load_pdf("data/pdf")
print(f"Loaded {len(pdf_file)} documents")
# Split into chunks
chunks = split_documents(pdf_file)
print(f"split {len(pdf_file)} documents into {len(chunks)} chunks")

# Example of a chunk
if chunks:
    print(f"\nExample Chunks:")
    print(f"Content: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")

# -------Embedding Manager------
embedding_manager=EmbeddingManager()
texts=[doc.page_content for doc in chunks]
embeddings=embedding_manager.generate_embeddings(texts)
print("\n Example Embedding Vector:")
#print(embeddings[0][:20])

# --------Vector-Store-------
vector_store=VectorStore()
vector_store.add_documents(chunks, embeddings)

# --------Retriever-------
retriever=RagRetriever(vector_store, embedding_manager)
results=retriever.retrieve("Factors behind operation Radd_ul-Fassad",top_k=3)
for r in results:
    print(f"- Rank {r['rank']} | Score: {r['similarity_score']:.4f} | Content: {r['content'][:100]}...")


# -------Rag Pipeline------
rag=RagPipeline(retriever)
query="Factor behind operation radd-ul-fassad"
answer=rag.query(query,top_k=3)
print("\n Final Answer:",answer)