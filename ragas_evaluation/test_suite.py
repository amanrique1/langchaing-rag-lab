"""
Complete RAGAS test suite for RAG evaluation
"""
import logging
from typing import Any, Dict, List, Optional

from ragas_evaluation.config import EvaluationConfig
from ragas_evaluation.dataset_generator import GroundTruthDatasetGenerator
from ragas_evaluation.evaluator import RAGEvaluator
from ragas_evaluation.metrics_analyzer import MetricsAnalyzer
from ragas_evaluation.utils import timing_decorator

from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.talk_use_case import TalkUseCase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGASTestSuite:
    """Complete test suite for RAG system evaluation"""

    def __init__(
        self,
        storage_handler: StorageUseCase,
        talk_use_case: TalkUseCase,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize test suite

        Args:
            storage_handler: StorageUseCase instance
            talk_use_case: TalkUseCase instance
            config: Evaluation configuration
        """
        self.storage_handler = storage_handler
        self.talk_use_case = talk_use_case
        self.config = config or EvaluationConfig()

        self.evaluator = RAGEvaluator(self.config, self.talk_use_case)
        self.analyzer = MetricsAnalyzer(self.config)
        self.dataset_generator = GroundTruthDatasetGenerator()

        logger.info("✓ RAGASTestSuite initialized")

    @timing_decorator
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline

        Returns:
            Complete evaluation results
        """
        print("\n" + "🚀 "*30)
        print("STARTING COMPLETE RAG EVALUATION")
        print("🚀 "*30)

        # Step 1: Generate ground truth dataset
        print("\n📝 Step 1: Generating Ground Truth Dataset")
        ground_truth_data = self.dataset_generator.create_comprehensive_dataset()
        print(f"✓ Generated {len(ground_truth_data)} test questions")

        # Step 2: Create evaluation dataset
        print("\n🔄 Step 2: Querying RAG System")
        dataset = self.evaluator.create_evaluation_dataset(
            ground_truth_data,
            self.storage_handler
        )

        # Step 3: Run RAGAS evaluation
        print("\n📊 Step 3: Running RAGAS Evaluation")
        eval_results = await self.evaluator.evaluate(dataset)

        # Step 4: Analyze results
        print("\n🔍 Step 4: Analyzing Results")
        analysis = self.analyzer.analyze_results(eval_results["results"])

        # Step 5: Display summary
        self.analyzer.display_summary(analysis)

        # Step 6: Export results
        if self.config.save_results:
            print("\n💾 Step 5: Exporting Results")
            self.analyzer.export_results(
                analysis,
                dataset,
                self.config.results_dir
            )

        # Compile complete results
        complete_results = {
            "evaluation": eval_results,
            "analysis": analysis,
            "dataset": dataset,
            "config": self.config.__dict__
        }

        print("\n" + "✅ "*30)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("✅ "*30 + "\n")

        return complete_results

    @timing_decorator
    async def run_category_evaluation(self) -> Dict[str, Any]:
        """Run evaluation with category breakdown"""
        print("\n📂 Running Category-Based Evaluation")

        ground_truth_data = self.dataset_generator.create_comprehensive_dataset()
        dataset = self.evaluator.create_evaluation_dataset(
            ground_truth_data,
            self.storage_handler
        )

        results = await self.evaluator.evaluate_with_categories(dataset)

        # Analyze each category
        category_analysis = {}
        for category, cat_results in results.items():
            analysis = self.analyzer.analyze_results(cat_results["results"])
            category_analysis[category] = analysis

            print(f"\n{'='*60}")
            print(f"Category: {category.upper()}")
            print(f"{'='*60}")
            self.analyzer.display_summary(analysis)

        return {
            "category_results": results,
            "category_analysis": category_analysis
        }

    async def run_quick_test(self, num_samples: int = 5) -> Dict[str, Any]:
        """Run quick test with limited samples"""
        print(f"\n⚡ Running Quick Test ({num_samples} samples)")

        ground_truth_data = self.dataset_generator.create_comprehensive_dataset()[:num_samples]
        dataset = self.evaluator.create_evaluation_dataset(
            ground_truth_data,
            self.storage_handler
        )

        eval_results = await self.evaluator.evaluate(dataset)
        analysis = self.analyzer.analyze_results(eval_results["results"])
        self.analyzer.display_summary(analysis)

        return {
            "evaluation": eval_results,
            "analysis": analysis
        }

    def compare_configurations(
        self,
        configurations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple RAG configurations"""
        print("\n🔬 Running Configuration Comparison")

        comparison_results = {}

        for idx, config_params in enumerate(configurations):
            print(f"\n{'─'*60}")
            print(f"Testing Configuration {idx + 1}/{len(configurations)}")
            print(f"Parameters: {config_params}")
            print(f"{'─'*60}")

            # Update configuration and run evaluation
            # This would require passing config params to RAG system
            results = self.run_quick_test(num_samples=10)
            comparison_results[f"config_{idx}"] = results

        return comparison_results