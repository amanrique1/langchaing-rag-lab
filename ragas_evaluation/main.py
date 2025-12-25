import asyncio
from ragas_evaluation.test_suite import RAGASTestSuite
from ragas_evaluation.config import EvaluationConfig
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.chunking_use_case import ChunkingUseCase
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.models.google_genai_language_model import GoogleGenAILanguageModel
from src.infrastructure.adapters.models.google_genai_embedding_model import GoogleGenAIEmbeddingModel
from src.infrastructure.adapters.document_loaders.markdown_loader import MarkdownDocumentLoader
from src.domain.models.config_classes import ChunkingConfig

async def main():
    # Custom configuration
    config = EvaluationConfig(
        metrics_to_evaluate=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        thresholds={
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.85,
            "context_recall": 0.80
        },
        enable_detailed_logging=True
    )

    # Instantiate shared models
    language_model = GoogleGenAILanguageModel()
    embedding_model = GoogleGenAIEmbeddingModel()
    document_loader = MarkdownDocumentLoader()

    # Create use cases with orchestration layer
    storage_use_case = StorageUseCase(
        collection_name="ragas_evaluation_store",
        embedding_model=embedding_model,
        use_ensemble=False  # Disable ensemble for evaluation
    )
    
    talk_use_case = TalkUseCase(
        collection_name="ragas_evaluation_store",
        embedding_model=embedding_model,
        language_model=language_model,
        use_ensemble=False  # Disable ensemble for evaluation
    )
    
    chunking_use_case = ChunkingUseCase(document_loader)

    # Prepare and run chunking
    chunk_config = ChunkingConfig(
        source_path="data",
        strategy="semantic",
        strategy_config={}
    )
    
    storage_use_case.clear()
    chunks = chunking_use_case.execute(
        source=chunk_config.source_path,
        strategy_name=chunk_config.strategy,
        strategy_config=chunk_config.strategy_config
    )
    storage_use_case.save(chunks)

    # Initialize suite
    suite = RAGASTestSuite(storage_use_case, talk_use_case, config)

    # Run full evaluation
    full_results = await suite.run_full_evaluation()

    # Run category-based evaluation
    category_results = await suite.run_category_evaluation()

    # Quick test
    quick_results = await suite.run_quick_test(num_samples=5)

def run():
    """Synchronous entry point function for Poetry scripts."""
    asyncio.run(main())

if __name__ == "__main__":
    run()