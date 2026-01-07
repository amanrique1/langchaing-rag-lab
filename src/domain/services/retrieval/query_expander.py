import json
import logging
import time
from pathlib import Path
from typing import Union, List

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
            # Determine template filename
            is_multi_query = self.strategy in (QueryExpansionStrategy.ZERO_SHOT, QueryExpansionStrategy.FEW_SHOT)
            template_filename = "multi_query.txt" if is_multi_query else f"{self.strategy.value}.txt"
            template_path = self.TEMPLATE_BASE_PATH / template_filename

            # Validate and load template
            if not template_path.exists():
                raise ValueError(
                    f"Template file not found: {template_path}\n"
                    f"Strategy: {self.strategy.value}"
                )

            try:
                self._template = template_path.read_text(encoding='utf-8')
                logger.debug(
                    f"Template loaded: {template_path.name} "
                    f"({len(self._template)} chars, strategy: {self.strategy.value})"
                )
            except Exception as e:
                raise ValueError(f"Failed to read template file {template_path}: {e}")

        return self._template

    def generate(self, question: str) -> Union[str, List[str]]:
        """
        Generates an augmented query for the given question using the specified strategy.

        Args:
            question (str): The question to generate an augmented query for.

        Returns:
            Union[str, List[str]]:
                - str for single-output strategies (HYDE, STEPBACK)
                - List[str] for multi-output strategies (SUBQUERIES, ZERO_SHOT, FEW_SHOT)

        Raises:
            ValueError: If few-shot examples can't be loaded or are invalid.
        """
        # Input validation
        if not question or not question.strip():
            logger.warning("Empty question received for query expansion")

        logger.debug(f"Generating queries with {self.strategy.value} strategy")
        start_time = time.time()

        # Build prompt based on strategy
        if self.strategy == QueryExpansionStrategy.FEW_SHOT:
            # Load few-shot examples
            examples_path = self.TEMPLATE_BASE_PATH / "few_shot_examples.json"

            if not examples_path.exists():
                raise ValueError(f"Few-shot examples file not found: {examples_path}")

            try:
                examples = json.loads(examples_path.read_text(encoding='utf-8'))
                logger.debug(f"Loaded {len(examples)} few-shot examples")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {examples_path}: {e}")
            except Exception as e:
                raise ValueError(f"Failed to read {examples_path}: {e}")

            # Build few-shot prompt
            example_prompt = ChatPromptTemplate.from_messages([
                ("human", "{question}"),
                ("ai", "{answer}"),
            ])

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=examples,
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.template),
                few_shot_prompt,
                ("human", "{question}"),
            ])
        else:
            # Standard prompt (HYDE, STEPBACK, SUBQUERIES, ZERO_SHOT)
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.template),
                ("human", "{question}"),
            ])

        # Determine parser
        parser = NumberedListOutputParser() if self.strategy in self.LIST_BASED_STRATEGIES else None

        # Format prompt and generate answer
        try:
            formatted_prompt = prompt.format(question=question)

            llm_start_time = time.time()
            result = self.llm.get_answer(formatted_prompt, parser=parser)
            llm_duration = time.time() - llm_start_time

            total_duration = time.time() - start_time

            # Log internal metrics
            result_count = len(result) if isinstance(result, list) else 1
            logger.debug(
                f"Query generation complete: {result_count} variant(s) in {total_duration:.2f}s "
                f"(LLM: {llm_duration:.2f}s, strategy: {self.strategy.value})"
            )

            return result

        except Exception as e:
            logger.error(
                f"Query generation failed (strategy: {self.strategy.value}): {e}",
                exc_info=True
            )
            raise