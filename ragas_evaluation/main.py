import asyncio
from ragas_evaluation.test_suite import RAGASTestSuite
from ragas_evaluation.config import EvaluationConfig
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.chunking_use_case import ChunkingUseCase
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.language_models.google_genai_language_model import GoogleGenAILanguageModel
from src.infrastructure.adapters.language_models.google_genai_embedding_model import GoogleGenAIEmbeddingModel
from src.infrastructure.adapters.document_loaders.markdown_loader import MarkdownDocumentLoader
from src.domain.models.cli_config_classes import ChunkingConfig

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

    # Instantiate adapters
    language_model = GoogleGenAILanguageModel()
    embedding_model = GoogleGenAIEmbeddingModel()
    document_loader = MarkdownDocumentLoader()
    chunk_store = ChromaChunkStore("ragas_evaluation_store", embedding_model)

    # Instantiate use cases
    storage_use_case = StorageUseCase(chunk_store)
    talk_use_case = TalkUseCase(language_model, chunk_store)
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