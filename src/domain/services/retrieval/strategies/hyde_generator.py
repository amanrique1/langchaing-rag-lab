from langchain_core.prompts import ChatPromptTemplate
from src.application.ports.language_model import LanguageModel


class HyDEGenerator:
    """
    A generator that uses the HyDE (Hypothetical Document Embeddings) approach 
    to generate hypothetical answers.
    """
    HYDE_TEMPLATE = """You are an expert assistant. Write a brief, plausible, 
but hypothetical passage that answers the following question. 
Do not include any preamble or explanation, just the passage.

Question: {question}
Hypothetical Answer:"""
    
    def __init__(self, llm: LanguageModel):
        """
        Initializes the HyDE generator.
        
        Args:
            llm (LanguageModel): The language model to use for generating hypothetical answers.
        """
        self.llm = llm
    
    def generate(self, question: str) -> str:
        """
        Generates a hypothetical answer to the given question using the HyDE approach.
        
        Args:
            question (str): The question to generate a hypothetical answer for.
        
        Returns:
            str: The hypothetical answer to the question.
        """
        prompt = ChatPromptTemplate.from_template(self.HYDE_TEMPLATE)
        return self.llm.get_answer(prompt.format(question=question))