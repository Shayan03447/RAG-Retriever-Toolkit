from typing import List, Dict, Any, Tuple
import chromadb
import numpy as np
import uuid
import os

class VectorStore:
    """Manage documents embeddings in ChromaDB Vector_Store"""
    def __init__(self, collection_name: str="pdf_documents", presist_directory: str="data/vector_store"):
        self.collection_name=collection_name
        self.presist_directory=presist_directory
        self.client=None
        self.collection=None
        self._initialize_store()

    def _initialize_store(self):
        """Initialized ChromaaDB client and collection"""
        try:
            os.makedirs(self.presist_directory, exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.presist_directory)
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"Discription":"PDF documents embedding for RAG"})
            print(f"VectorStore Initialized collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error while initializing vector_store: {e} ")
            raise
    
    def add_documents(self, documents:List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store
        Args:
            Documents: List of langchain documents
            Embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")
        print(f"Adding {len(documents)} documents to vector store-----")

        # Prepare data for the Chromaa DB
        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate Unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            # Prepare Metadata
            metadata=dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            # Document content
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        # ADD TO COLLECTIONS
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added{len(documents)} documents to Vector Store")
            print(f"Total Documents in Collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error while adding documents to Vector Store: {e}")
            raise





        