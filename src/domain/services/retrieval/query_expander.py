import json
import logging
import time
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import NumberedListOutputParser

from src.application.ports.language_model import LanguageModel
from src.domain.models.enums import QueryExpansionStrategy

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Service for query augmentation strategies.
    Loads templates dynamically based on strategy type.

    Supported strategies:
    - HYDE: Generates hypothetical document
    - STEPBACK: Creates broader question
    - SUBQUERIES: Breaks down into multiple queries
    - ZERO_SHOT: Multi-query generation without examples
    - FEW_SHOT: Multi-query generation with examples
    """

    TEMPLATE_BASE_PATH = Path("assets/templates/query_augmentation")

    # Strategies that return lists instead of single strings
    LIST_BASED_STRATEGIES = {
        QueryExpansionStrategy.SUBQUERIES,
        QueryExpansionStrategy.ZERO_SHOT
    }

    def __init__(self, llm: LanguageModel, strategy: QueryExpansionStrategy):
        """
        Initializes the query augmentation service.

        Args:
            llm (LanguageModel): The language model to use for generating augmented queries.
            strategy (QueryExpansionStrategy): The augmentation strategy to use.
        """
        self.llm = llm
        self.strategy = strategy
        self._template: str = ""

        logger.debug(
            f"QueryExpander initialized: strategy={strategy.value}, llm={llm.__class__.__name__}"
        )

    @property
    def template(self) -> str:
        """
        Lazy-loads and returns the template for the current strategy.

        Returns:
            str: The template content.

        Raises:
            ValueError: If template file doesn't exist or can't be read.
        """
        if not self._template:
            is_multi_query = self.strategy in (
                QueryExpansionStrategy.ZERO_SHOT,
                QueryExpansionStrategy.FEW_SHOT
            )
            template_filename = "multi_query.txt" if is_multi_query else f"{self.strategy.value}.txt"
            template_path = self.TEMPLATE_BASE_PATH / template_filename

            if not template_path.exists():
                raise ValueError(f"Template file not found: {template_path}")

            self._template = template_path.read_text(encoding='utf-8')
            logger.debug(f"Template loaded: {template_path.name}")

        return self._template

    def expand(self, question: str) -> List[str]:
        """
        Expands a query into multiple variations.

        Args:
            question: The original query to expand

        Returns:
            List[str]: Expanded query variations (always includes original)
        """
        if not question or not question.strip():
            logger.warning("Empty question received")
            return [question]

        logger.debug(f"Expanding query with {self.strategy.value} strategy")
        start_time = time.time()

        try:
            # Build prompt based on strategy
            if self.strategy == QueryExpansionStrategy.FEW_SHOT:
                prompt = self._build_few_shot_prompt()
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.template),
                    ("human", "{question}"),
                ])

            # Determine parser
            parser = (NumberedListOutputParser()
                     if self.strategy in self.LIST_BASED_STRATEGIES
                     else None)

            # Generate
            formatted_prompt = prompt.format(question=question)
            result = self.llm.get_answer(formatted_prompt, parser=parser)

            # Normalize to list
            if isinstance(result, str):
                expanded = [result]
            else:
                expanded = result

            # Clean and validate
            expanded = [q.strip() for q in expanded if q and q.strip()]

            if not expanded:
                logger.warning("Expansion produced no results, using original")
                return [question]

            duration = time.time() - start_time
            logger.info(
                f"Query expanded: {len(expanded)} variant(s) in {duration:.2f}s "
                f"(strategy: {self.strategy.value})"
            )

            return expanded

        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            return [question]  # Fallback to original

    def _build_few_shot_prompt(self) -> ChatPromptTemplate:
        """Build few-shot prompt with examples."""
        examples_path = self.TEMPLATE_BASE_PATH / "few_shot_examples.json"

        if not examples_path.exists():
            raise ValueError(f"Few-shot examples not found: {examples_path}")

        examples = json.loads(examples_path.read_text(encoding='utf-8'))

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        return ChatPromptTemplate.from_messages([
            ("system", self.template),
            few_shot_prompt,
            ("human", "{question}"),
        ])
