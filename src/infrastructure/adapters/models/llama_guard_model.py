from pathlib import Path

from langchain_ollama import OllamaLLM
from src.application.ports.guardrail_model import GuardrailModel


class LlamaGuard(GuardrailModel):
    def __init__(self, model_id: str = "llama-guard3:1b"):
        """
        Initializes the Llama Guard model.

        Args:
            model_id (str): The Hugging Face model ID for Llama Guard.
        """
        self.model = OllamaLLM(model=model_id)

    def validate(self, text: str) -> bool:
        # Explicit instructions for the model
        template_path = Path("assets/templates/guard.txt")
        template = template_path.read_text()
        prompt = template.format(text=text)

        response = self.model.invoke(prompt)
        return "unsafe" not in response.lower()