from typing import List
from pathlib import Path
from src.application.ports.language_model import LanguageModel
from src.domain.models.chunk import Chunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

try:
    data_path = Path("assets/query_template.txt")
    if not data_path.exists():
        raise FileNotFoundError("The file assets/query_template.txt does not exist")
    QUERY_TEMPLATE_CONTENT = data_path.read_text()
except FileNotFoundError as e:
    # Handle the error appropriately, perhaps with a default template or a clearer startup error
    print(f"FATAL: Could not load query template. {e}")
    raise


class GoogleGenAILanguageModel(LanguageModel):
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        # The chain is the core of your use case. Define it once.
        # This is a sequence of operations: prompt -> model -> output_parser
        self.chain: Runnable = (
            ChatPromptTemplate.from_template(QUERY_TEMPLATE_CONTENT)
            | ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            | StrOutputParser()
        )

    def get_answer(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Generates an answer to a question based on the provided context chunks.

        Args:
            question (str): The question to be answered.
            context_chunks (List[Chunk]): A list of context chunks to be used for answering the question.

        Returns:
            str: The generated answer.
        """
        context = "\n\n".join([chunk.content for chunk in context_chunks])

        # The chain is invoked with a dictionary matching the variables in the template
        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        return response
