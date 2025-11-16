import asyncio
from ragas_evaluation.test_suite import RAGASTestSuite
from ragas_evaluation.config import EvaluationConfig
from src.application.use_cases.storage_use_case import StorageUseCase
from src.domain.models.enums import StorageType
from src.infrastructure.cli.main import run_chunking as init_env
from src.domain.models.cli_config_classes import ChunkingConfig, StorageConfig

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

    chunk_config = ChunkingConfig(
                    source_path="data",
                    strategy="semantic",
                    strategy_config={}
                )
    store_config = StorageConfig(
                    storage_type = StorageType.CHROMA,
                    location = "ragas_evaluation_store"
                )
    init_env(chunk_config, store_config)
    storage_handler = StorageUseCase(store_config.storage_type, store_config.location)

    # Initialize suite
    suite = RAGASTestSuite(storage_handler, config)

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