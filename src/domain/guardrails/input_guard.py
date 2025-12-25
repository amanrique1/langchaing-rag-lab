import re
from pathlib import Path
from typing import Optional
from src.domain.guardrails.config import GuardrailConfig
from src.application.ports.guardrail_model import GuardrailModel
from src.domain.exceptions.security_violation_exception import SecurityViolationError

class InputGuard:
    def __init__(self, guardrail_model: GuardrailModel):
        """
        Args:
            guardrail_model: An instance of a class implementing GuardrailModel (e.g., LlamaGuard).
        """
        self.guard_model = guardrail_model
        self.config = GuardrailConfig()
        
        # 1. Pre-compile regex patterns for performance
        self.compiled_jailbreaks = [
            re.compile(p, re.IGNORECASE) for p in self.config.JAILBREAK_PATTERNS
        ]

        # 2. Load the external template into memory once
        self.prompt_template = self._load_template(self.config.TEMPLATE_PATH)

    def _load_template(self, path: Path) -> str:
        """Helper to safely read the external txt file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Security Template not found at: {path}")

    def _check_fast_rules(self, text: str) -> None:
        """
        Layer 1: Regex Check. Raises exception if matched.
        """
        for pattern in self.compiled_jailbreaks:
            if pattern.search(text):
                msg = f"Blocked by Fast Rule: Pattern '{pattern.pattern}' detected."
                print(f"xx [Input Guard] {msg}")
                raise SecurityViolationError(msg, violation_type="REGEX")

    def _check_semantic_intent(self, text: str) -> None:
        """
        Layer 2: AI Model Check. Raises exception if unsafe.
        """
        is_safe = self.guard_model.validate(text)
        if not is_safe:
            msg = "Blocked by Semantic Guardrail (Llama Guard)."
            print(f"xx [Input Guard] {msg}")
            raise SecurityViolationError(msg, violation_type="LLAMA_GUARD")

    def _validate(self, text: str) -> None:
        """
        Executes validation layers. Raises SecurityViolationError on failure.
        """
        self._check_fast_rules(text)
        self._check_semantic_intent(text)

    def build_safe_query(self, query: str, context: str) -> Optional[str]:
        """
        Performs validation and constructs the final secure prompt.

        Args:
            query (str): The user's question.
            context (str): The retrieved technical documentation.

        Returns:
            str: The fully formatted prompt ready for the LLM.
            None: If the query was deemed unsafe.
        """
        # 1. Validate the User Query (we usually don't validate the trusted context)
        self._validate(query)

        # 2. Fill the external template
        # We use .format() to inject the dynamic data into the txt content
        try:
            safe_prompt = self.prompt_template.format(
                context=context,
                question=query
            )
            return safe_prompt
        except KeyError as e:
            print(f"!! Template Error: Missing placeholder {e} in text file.")
            return None