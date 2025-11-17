import pytest
from src.application.ports.language_model import LanguageModel
from src.domain.models.chunk import Chunk


def test_abc_methods_can_be_called_via_super():
    """
    This test ensures 100% coverage of the ABC by creating a subclass
    that explicitly calls the super() method for each abstract method.
    This executes the 'pass' statements in the LanguageModel ABC.
    """

    class SuperCallingLanguageModel(LanguageModel):
        def get_answer(self, question: str, context_chunks: list[Chunk]) -> str:
            return super().get_answer(question, context_chunks)

    # Instantiate and call each method to hit the 'pass' lines in the ABC
    super_model = SuperCallingLanguageModel()
    super_model.get_answer("some_question", [])
    # No assertions needed, the goal is simply to execute the code.
