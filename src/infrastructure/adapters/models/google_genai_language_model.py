from typing import List
from src.application.ports.language_model import LanguageModel
from src.domain.guardrails.input_guard import InputGuard
from src.domain.models.chunk import Chunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

class GoogleGenAILanguageModel(LanguageModel):
    def __init__(self, guard: InputGuard, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        # The chain is the core of your use case. Define it once.
        # This is a sequence of operations: prompt -> model -> output_parser
        self.guard = guard
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.parser = StrOutputParser()


    def get_answer(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Generates an answer to a question based on the provided context chunks.

        Args:
            question (str): The question to be answered.
            context_chunks (List[Chunk]): A list of context chunks to be used for answering the question.

        Returns:
            str: The generated answer.
        """
        context_text = "\n\n".join([chunk.content for chunk in context_chunks])

        try:
            # 1. Attempt to build the safe query
            # This will raise SecurityViolationError if the input is malicious
            safe_prompt_string = self.guard.build_safe_query(question, context_text)

            # 2. Invoke the chain with the validated prompt
            response = (self.model | self.parser).invoke(safe_prompt_string)
            return response

        except SecurityViolationError as e:
            # 3. Handle Security Violations specifically
            # Log the specific violation type (REGEX vs MODEL) for auditing
            print(f"SECURITY ALERT [{e.violation_type}]: {e}")
            
            # Return a polite refusal to the user (don't crash the app)
            return (
                "I cannot fulfill this request as it violates our security policies "
                "regarding safe and appropriate content."
            )

        except Exception as e:
            # 4. Handle other runtime errors (network, template issues, etc.)
            print(f"SYSTEM ERROR: {e}")
            return "An unexpected error occurred while processing your request."