import logging
from src.application.ports.language_model import LanguageModel
from src.application.use_cases.search_use_case import SearchUseCase
from src.domain.guardrails.input_guard import InputGuard
from src.domain.exceptions.security_violation_exception import SecurityViolationError

logger = logging.getLogger(__name__)

class TalkUseCase:
    """
    Orchestrates the 'Chat with your Data' workflow.
    
    This use case composes the SearchUseCase to retrieve relevant documentation
    and uses the LanguageModel to generate a grounded answer.
    """
    
    def __init__(
        self,
        language_model: LanguageModel,
        search_use_case: SearchUseCase,
        input_guard: InputGuard
    ):
        """
        Initialize the Talk Use Case with injected dependencies.

        Args:
            language_model (LanguageModel): The generative model responsible for answering.
            search_use_case (SearchUseCase): The retrieval use case.
            input_guard (InputGuard): The security gateway for validation and grounding.
        """
        self.language_model = language_model
        self.search_use_case = search_use_case
        self.input_guard = input_guard
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> str:
        """
        Executes the question-answering pipeline synchronously.

        1. Delegates retrieval to SearchUseCase.
        2. Checks if valid context exists.
        3. Delegates generation to LanguageModel.

        Args:
            query (str): The user's natural language question.
            top_k (int): Number of chunks to provide as context to the LLM.
            num_candidates (int): Number of chunks to retrieve before reranking.
            use_reranking (bool): Whether to enable the reranker in the search step.
            
        Returns:
            str: The generated answer or a fallback message if no context is found.
        """
        # 1. Retrieve Context (Reuse logic from SearchUseCase)
        chunks = self.search_use_case.execute(
            query=query, 
            top_k=top_k, 
            num_candidates=num_candidates, 
            use_reranking=use_reranking
        )
        
        # 2. Early Exit if no context
        if not chunks:
            return "I could not find any relevant information in the documentation to answer your question."

        # 3. Security Grounding & Answer Generation
        try:
            # Aggregate context
            context_text = "\n\n".join([chunk.content for chunk in chunks])
            
            # Construct Safe Prompt (Performs Regex + LlamaGuard validation)
            safe_prompt = self.input_guard.build_safe_query(query, context_text)
            
            if safe_prompt is None:
                return "An internal error occurred while preparing the secure prompt."

            return self.language_model.get_answer(safe_prompt)

        except SecurityViolationError as e:
            logger.warning(f"Security Alert [{e.violation_type}]: {str(e)}")
            return (
                "I cannot fulfill this request as it violates our security policies "
                "regarding safe and appropriate content."
            )
        except Exception as e:
            logger.error(f"Error in TalkUseCase generation: {e}", exc_info=True)
            return "An unexpected error occurred while generating the answer."

    async def aexecute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> str:
        """
        Executes the question-answering pipeline asynchronously.
        
        Recommended for web/API implementations to prevent blocking the event loop
        during the LLM generation phase.

        Args:
            query (str): The user's natural language question.
            top_k (int): Number of chunks to provide as context.
            num_candidates (int): Number of candidates for reranking.
            use_reranking (bool): Whether to enable reranking.

        Returns:
            str: The generated answer.
        """
        # 1. Retrieve Context
        chunks = self.search_use_case.execute(
            query=query, 
            top_k=top_k, 
            num_candidates=num_candidates, 
            use_reranking=use_reranking
        )
        # 2. Early Exit if no context
        if not chunks:
            return "I could not find any relevant information in the documentation to answer your question."

        # 3. Security Grounding & Answer Generation
        try:
            # Aggregate context
            context_text = "\n\n".join([chunk.content for chunk in chunks])
            
            # Construct Safe Prompt (Performs Regex + LlamaGuard validation)
            safe_prompt = self.input_guard.build_safe_query(query, context_text)
            
            if safe_prompt is None:
                return "An internal error occurred while preparing the secure prompt."

            return await self.language_model.aget_answer(safe_prompt)

        except SecurityViolationError as e:
            logger.warning(f"Security Alert [{e.violation_type}]: {str(e)}")
            return (
                "I cannot fulfill this request as it violates our security policies "
                "regarding safe and appropriate content."
            )
        except Exception as e:
            logger.error(f"Error in TalkUseCase generation (async): {e}", exc_info=True)
            return "An unexpected error occurred while generating the answer."