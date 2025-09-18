from  typing import List, Tuple, Any, Dict
from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager

class RagRetriever:
    """Hanlde query based retrievel from vector-store"""
    
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):

        """
        Initialize the retriever
        Basicallly retriever is actually built on the top of vector_store

        args:
            Vectorstore: Where the embeddings are stored
            Embedding_manager: Convert the query into embeddings
        """
        self.vector_store=vectorstore 
        self.embedding_manager=embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float= 0.0) -> List[Dict[str,Any]]:
        """
        Retrieve top_k relevant documents for the given query using the vector store.
        Args:
            Query: The search query
            Top_K: Number of results return for the query from vectorstore
            score_threshold:Minimun similarity score threshold
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top_k: {top_k}, score_threshold: {score_threshold}")
        # Generate query embedding 
        query_embedding=self.embedding_manager.generate_embeddings([query])[0]
        # Search in Vector-Store
        try:
            results=self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents','metadatas','distances']
            )
            # Process Result 
            retrieved_docs=[]
            if results['documents'] and results['documents'][0]:
                documents=results['documents'][0]
                metadatas=results['metadatas'][0]
                distances=results['distances'][0]
                
                for i, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score(ChromaDB uses cosine distnace)
                    similarity_score = 1/(1+distance)
                    # if similarity_score >= score_threshold:
                    if True:
                        retrieved_docs.append({
                            "id": metadata.get("doc_index",i),
                            "content": document,
                            "metadata":metadata,
                            "similarity_score":similarity_score,
                            "distance":distance,
                            "rank":i+1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No document found")
            
            return retrieved_docs
        except Exception as e:
            print(f"[Retriever] Error: {e}")

        