import torch
from typing import List
from src.application.ports.embedding_model import EmbeddingModel
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        super().__init__(model_name=model_name)
        model_kwargs = {"device": self._detect_device()}
        encode_kwargs = {"normalize_embeddings": True}
        self.embedding_model = HuggingFaceEmbeddings(model=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The generated embedding.
        """
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The list of texts to be embedded.

        Returns:
            List[List[float]]: A list of embeddings for the given texts.
        """
        return self.embedding_model.embed_documents(texts)

    def _detect_device(self) -> str:
        """Helper to determine the best available hardware acceleration."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"
