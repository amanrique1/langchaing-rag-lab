from langchain_core.prompts import ChatPromptTemplate
from src.application.ports.language_model import LanguageModel


class StepBackGenerator:
    """
    A generator that uses the Step-Back prompting approach to generate 
    broader, more general questions for better retrieval.
    """
    STEPBACK_TEMPLATE = """You are an expert at world knowledge. Your task is to step back 
    and paraphrase a question to a more generic step-back question, which is easier to answer.

    Here are a few examples:

    Original Question: What happens to the pressure, P, of an ideal gas if the temperature is 
    increased by a factor of 2 and the volume is increased by a factor of 8?
    Step-back Question: What are the physics principles behind the ideal gas law?

    Original Question: Who was the spouse of Anna Karina from 1968 to 1974?
    Step-back Question: Who were the spouses of Anna Karina?

    Original Question: {question}
    Step-back Question:"""
    
    def __init__(self, llm: LanguageModel):
        """
        Initializes the Step-Back generator.
        
        Args:
            llm (LanguageModel): The language model to use for generating step-back questions.
        """
        self.llm = llm
    
    def generate(self, question: str) -> str:
        """
        Generates a step-back question for the given question.
        
        Args:
            question (str): The question to generate a step-back question for.
        
        Returns:
            str: The step-back question.
        """
        prompt = ChatPromptTemplate.from_template(self.STEPBACK_TEMPLATE)
        return self.llm.get_answer(prompt.format(question=question))