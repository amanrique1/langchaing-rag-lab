from typing import List
from src.application.ports.embedding_model import EmbeddingModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GoogleGenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "models/embedding-001"):
        super().__init__(model_name=model_name)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

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
