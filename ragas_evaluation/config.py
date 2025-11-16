"""
Configuration module for RAGAS evaluation suite
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

class ModelProvider(Enum):
    GOOGLE = "google"
    OLLAMA = "ollama"

@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation"""

    # Model Configuration
    model_provider: ModelProvider = ModelProvider.GOOGLE
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"
    temperature: float = 0.0

    # Evaluation Settings
    metrics_to_evaluate: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ])

    # Performance Thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.8,
        "answer_relevancy": 0.75,
        "context_precision": 0.8,
        "context_recall": 0.75,
        "overall": 0.75
    })

    # Dataset Configuration
    num_test_samples: int = 20
    include_edge_cases: bool = True

    # Output Configuration
    save_results: bool = True
    results_dir: str = "./evaluation_results"
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])

    # Advanced Settings
    enable_detailed_logging: bool = True
    parallel_evaluation: bool = False
    retry_failed_queries: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration after initialization"""
        os.makedirs(self.results_dir, exist_ok=True)

        # Validate thresholds
        for metric, threshold in self.thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"Threshold for {metric} must be between 0 and 1")