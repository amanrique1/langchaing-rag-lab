"""
Core RAG evaluation engine using RAGAS
"""
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import logging
from tqdm import tqdm

from ragas_evaluation.config import EvaluationConfig
from ragas_evaluation.utils import RetryHandler, ProgressTracker
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.storage_use_case import StorageUseCase

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Comprehensive RAG evaluation using RAGAS metrics"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = None
        self.embeddings = None
        self.retry_handler = RetryHandler(max_retries=config.max_retries)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        try:
            if self.config.model_provider.value == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.llm_model,
                    temperature=self.config.temperature
                )
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.config.embedding_model
                )
                self.chat_handler = TalkUseCase()
                logger.info("✓ Google AI models initialized")
            else:
                raise NotImplementedError(
                    f"Provider {self.config.model_provider} not implemented"
                )
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def create_evaluation_dataset(
        self,
        ground_truth_data: List[Dict[str, Any]],
        storage_handler: StorageUseCase
    ) -> List[Dict[str, Any]]:
        """
        Create evaluation dataset by querying the RAG system

        Args:
            ground_truth_data: List of questions with ground truth answers
            vector_store: Vector store instance
            ask_rag_function: Function to query the RAG system

        Returns:
            List of dataset entries with contexts and generated answers
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION DATASET")
        print("="*60)

        dataset = []
        progress = ProgressTracker(total=len(ground_truth_data))

        for item in tqdm(ground_truth_data, desc="Processing questions"):
            try:
                question = item["question"]
                ground_truth = item["ground_truth"]

                # Query RAG system with retry logic
                relevant_chunks = self.retry_handler.execute(
                    handler = storage_handler,
                    method_name = "search",
                    query=question
                )

                # Extract contexts
                contexts = [chunk.content for chunk in relevant_chunks]

                # Validate that we have contexts
                if not contexts:
                    logger.warning(f"No contexts retrieved for: {question[:50]}...")
                    contexts = ["No relevant context found"]

                answer = self.chat_handler.execute(
                    query=question,
                    relevant_chunks=relevant_chunks
                )

                dataset_entry = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "metadata": {
                        "category": item.get("category", "unknown"),
                        "difficulty": item.get("difficulty", "medium"),
                        "keywords": item.get("keywords", [])
                    }
                }

                dataset.append(dataset_entry)
                progress.update(success=True)

                if self.config.enable_detailed_logging:
                    logger.info(f"✓ Processed: {question[:50]}...")

            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}")
                progress.update(success=False)

                if not self.config.retry_failed_queries:
                    raise

        print(f"\n✓ Dataset created with {len(dataset)} examples")
        progress.display_summary()

        return dataset

    async def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS metrics

        Args:
            dataset: Evaluation dataset

        Returns:
            Evaluation results
        """
        print("\n" + "="*60)
        print("RAGAS EVALUATION")
        print("="*60)

        # Prepare metrics
        metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall
        }

        selected_metrics = [
            metrics_map[m] for m in self.config.metrics_to_evaluate
            if m in metrics_map
        ]

        # Convert to RAGAS format
        dataset_dict = {
            "question": [item["question"] for item in dataset],
            "answer": [item["answer"] for item in dataset],
            "contexts": [item["contexts"] for item in dataset],
            "ground_truth": [item["ground_truth"] for item in dataset]
        }

        eval_dataset = Dataset.from_dict(dataset_dict)

        print(f"\n📊 Evaluating {len(dataset)} questions...")
        print(f"📈 Metrics: {', '.join(self.config.metrics_to_evaluate)}")

        start_time = time.time()

        try:
            results = evaluate(
                dataset=eval_dataset,
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            elapsed_time = time.time() - start_time
            print(f"\n✓ Evaluation completed in {elapsed_time:.2f} seconds")

            return {
                "results": results,
                "dataset": dataset,
                "metadata": {
                    "num_samples": len(dataset),
                    "metrics": self.config.metrics_to_evaluate,
                    "elapsed_time": elapsed_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    async def evaluate_with_categories(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate with category breakdown

        Args:
            dataset: Evaluation dataset with metadata

        Returns:
            Results broken down by category
        """
        # Group by category
        categories = {}
        for item in dataset:
            category = item.get("metadata", {}).get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(item)

        # Evaluate each category
        category_results = {}
        for category, items in categories.items():
            print(f"\n{'─'*60}")
            print(f"Evaluating category: {category.upper()}")
            print(f"{'─'*60}")

            results = await self.evaluate(items)
            category_results[category] = results

        return category_results