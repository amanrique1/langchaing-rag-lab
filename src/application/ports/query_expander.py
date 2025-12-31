from abc import ABC, abstractmethod


class QueryExpander(ABC):
    """
    Port for query generation strategies (HyDE, StepBack, etc.)
    """
    
    @abstractmethod
    def generate(self, question: str) -> str:
        """
        Generate a query variation based on the original question.
        
        Args:
            question (str): The original question.
        
        Returns:
            str: The generated query variation.
        """
        pass