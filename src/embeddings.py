import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

class EmbeddingManager:
    """Handle documents embedding by the use of SentenceTransformer"""
    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        """Initialize the embedding_manager"""
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded successfully."
                f"Embedding Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error while Loading the model {self.model_name}: {e}")
            raise
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create Embeddings for a list of texts
        Texts: list of text string for embeddings

        Return: Numpy array pf embeddings with shape len(texts)
                embedding dimension
        """
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generate embeddings for {len(texts)} text---")
        embeddings=self.model.encode(texts, show_progress_bar=True)
        print(f"Generate embeddings for shape: {embeddings.shape}")
        return embeddings

            
        