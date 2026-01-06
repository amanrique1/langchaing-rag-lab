from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from src.application.ports.language_model import LanguageModel
from src.domain.models.enums import QueryExpansionStrategy


class QueryExpander:
    """
    Service for query augmentation strategies.
    Loads templates dynamically based on strategy type.
    """

    # Template mapping
    TEMPLATE_BASE_PATH: str = "assets/templates/query_augmentation/"

    def __init__(self, llm: LanguageModel, strategy: QueryExpansionStrategy):
        """
        Initializes the query augmentation service.

        Args:
            llm (LanguageModel): The language model to use for generating augmented queries.
        """
        self.llm = llm
        self.strategy = strategy
        self._template: str = ""

    @property
    def template(self) -> str:
        """
        Returns the template for the current strategy.
        """
        if not self._template:
            template_path = self.TEMPLATE_BASE_PATH + self.strategy.value + ".txt"
            if not Path(template_path).exists():
                raise ValueError(f"Template file not found: {template_path}")
            self._template = Path(template_path).read_text()
        return self._template

    def generate(self, question: str) -> str:
        """
        Generates an augmented query for the given question using the specified strategy.

        Args:
            question (str): The question to generate an augmented query for.

        Returns:
            str: The augmented query.
        """
        template = self.template
        prompt = ChatPromptTemplate.from_template(template)
        return self.llm.get_answer(prompt.format(question=question))