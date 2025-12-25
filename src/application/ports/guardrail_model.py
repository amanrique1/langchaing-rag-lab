from abc import ABC, abstractmethod


class GuardrailModel(ABC):
    """Guardrail model interface for validating input text."""
    
    @abstractmethod
    def validate(self, text: str) -> bool:
        """
        Validates the input text for any unsafe content.

        Args:
            text (str): The text to be validated.

        Returns:
            bool: True if the text is safe, False otherwise.
        """
        pass
