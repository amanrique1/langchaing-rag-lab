from src.application.ports.language_model import LanguageModel
from src.application.use_cases.search_use_case import SearchUseCase

class TalkUseCase:
    """
    Orchestrates the 'Chat with your Data' workflow.
    
    This use case composes the SearchUseCase to retrieve relevant documentation
    and uses the LanguageModel to generate a grounded answer.
    """
    
    def __init__(
        self,
        language_model: LanguageModel,
        search_use_case: SearchUseCase
    ):
        """
        Initialize the Talk Use Case with injected dependencies.

        Args:
            language_model (LanguageModel): The generative model (e.g., Gemini) responsible 
                                            for synthesizing the answer.
            search_use_case (SearchUseCase): The retrieval use case responsible for 
                                             finding, ranking, and extracting relevant chunks.
        """
        self.language_model = language_model
        self.search_use_case = search_use_case
    
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
        
        # 3. Generate Answer
        return self.language_model.get_answer(query, chunks)

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

        if not chunks:
            return "I could not find any relevant information in the documentation to answer your question."
        
        # 2. Generate Answer Asynchronously (Network bound)
        # Uses the aget_answer method we added to GoogleGenAILanguageModel
        return await self.language_model.aget_answer(query, chunks)