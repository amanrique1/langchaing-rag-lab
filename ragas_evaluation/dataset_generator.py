"""
Dataset generation module with multiple test scenarios
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class TestQuestion:
    """Structured test question with metadata"""
    question: str
    ground_truth: str
    category: str
    difficulty: str  # easy, medium, hard
    keywords: List[str]
    expected_doc_types: List[str] = None

class GroundTruthDatasetGenerator:
    """Generate comprehensive ground truth datasets for RAG evaluation"""

    def __init__(self, domain: str = "software_engineering"):
        self.domain = domain
        self.categories = []

    def create_comprehensive_dataset(self) -> List[Dict[str, Any]]:
        """Create a comprehensive dataset covering multiple scenarios"""

        datasets = [
            self._create_factual_questions(),
            self._create_conceptual_questions(),
            self._create_comparative_questions(),
            self._create_procedural_questions(),
            self._create_edge_cases(),
            self._create_ambiguous_questions()
        ]

        # Flatten all datasets
        all_questions = []
        for dataset in datasets:
            all_questions.extend(dataset)

        return all_questions

    def _create_factual_questions(self) -> List[Dict[str, Any]]:
        """Factual questions with specific answers"""
        return [
            {
                "question": "What are the fundamental principles of REST?",
                "ground_truth": "The fundamental principles of REST are: 1) Client-Server (clear separation of responsibilities), 2) Stateless (each request must contain all necessary information), 3) Cacheable (responses should be marked as cacheable or not), 4) Uniform Interface (consistent use of HTTP methods and URIs), and 5) Layered System (the architecture can be composed of multiple layers).",
                "category": "factual",
                "difficulty": "medium",
                "keywords": ["REST", "principles", "architecture"]
            },
            {
                "question": "What HTTP codes are used for client errors?",
                "ground_truth": "HTTP codes for client errors (4xx) include: 400 Bad Request (malformed request), 401 Unauthorized (authentication required), 403 Forbidden (access denied), 404 Not Found (resource not found), 409 Conflict (conflict with the current state of the resource), and 422 Unprocessable Entity (invalid input data).",
                "category": "factual",
                "difficulty": "easy",
                "keywords": ["HTTP", "status codes", "errors"]
            },
            {
                "question": "Which HTTP methods are idempotent?",
                "ground_truth": "The idempotent HTTP methods are GET (retrieve resources), PUT (update/create resources), and DELETE (delete resources). POST and PATCH are not idempotent.",
                "category": "factual",
                "difficulty": "medium",
                "keywords": ["HTTP", "idempotent", "methods"]
            }
        ]

    def _create_conceptual_questions(self) -> List[Dict[str, Any]]:
        """Questions requiring conceptual understanding"""
        return [
            {
                "question": "How does JWT authentication work?",
                "ground_truth": "JWT (JSON Web Tokens) has the structure Header.Payload.Signature. JWT tokens contain encoded and signed user information, allowing verification without consulting the database. It is implemented using a Bearer Token in the Authorization header: Bearer <token>. The header contains the algorithm (e.g., HS256) and type (JWT), while the payload includes data such as sub, name, iat, and exp.",
                "category": "conceptual",
                "difficulty": "medium",
                "keywords": ["JWT", "authentication", "security"]
            },
            {
                "question": "Explain the difference between authentication and authorization",
                "ground_truth": "Authentication is the process of verifying who a user is (validating identity), typically through credentials like username and password. Authorization is the process of verifying what specific applications, files, and data a user has access to (validating permissions). Authentication always comes before authorization.",
                "category": "conceptual",
                "difficulty": "medium",
                "keywords": ["authentication", "authorization", "security"]
            }
        ]

    def _create_comparative_questions(self) -> List[Dict[str, Any]]:
        """Questions requiring comparison of concepts"""
        return [
            {
                "question": "What are the advantages and disadvantages of microservices?",
                "ground_truth": "The advantages of microservices include: independent scalability, heterogeneous technologies, and independent deployment. The disadvantages are: network complexity, distributed data management, and complex monitoring.",
                "category": "comparative",
                "difficulty": "hard",
                "keywords": ["microservices", "architecture", "comparison"]
            },
            {
                "question": "Compare SQL and NoSQL databases",
                "ground_truth": "SQL databases are relational, use structured schemas, support ACID transactions, and are vertically scalable. They're ideal for complex queries and relationships. NoSQL databases are non-relational, have flexible schemas, offer eventual consistency, and are horizontally scalable. They're better for large-scale data and high performance requirements.",
                "category": "comparative",
                "difficulty": "hard",
                "keywords": ["SQL", "NoSQL", "databases"]
            }
        ]

    def _create_procedural_questions(self) -> List[Dict[str, Any]]:
        """Questions about procedures and processes"""
        return [
            {
                "question": "What are the rate limiting strategies?",
                "ground_truth": "Rate limiting strategies include: Fixed Window (fixed limit per time window), Sliding Window (more precise sliding window), Token Bucket (allows controlled bursts), and Leaky Bucket (constant flow of requests). It is implemented with headers like X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset.",
                "category": "procedural",
                "difficulty": "hard",
                "keywords": ["rate limiting", "API", "strategies"]
            },
            {
                "question": "What is the mandatory Git Flow model according to the policies?",
                "ground_truth": "The mandatory Git Flow model includes: main/master (production code), develop (integration of new functionalities), feature/* (development of new features), release/* (preparation of versions), and hotfix/* (urgent production fixes). The required format for branches is feature/TICKET-123-short-description.",
                "category": "procedural",
                "difficulty": "medium",
                "keywords": ["Git Flow", "branching", "version control"]
            },
            {
                "question": "What are the minimum requirements for Pull Requests?",
                "ground_truth": "The mandatory minimum requirements for Pull Requests are: a minimum of 2 approved reviewers, all automated tests successful, code coverage greater than 80%, no merge conflicts, and updated documentation if applicable.",
                "category": "procedural",
                "difficulty": "medium",
                "keywords": ["pull requests", "code review", "standards"]
            }
        ]

    def _create_edge_cases(self) -> List[Dict[str, Any]]:
        """Edge cases and corner scenarios"""
        return [
            {
                "question": "What happens when you send a PUT request without an ID?",
                "ground_truth": "A PUT request typically requires an ID to identify the resource to update. Without an ID, it may result in a 400 Bad Request or 404 Not Found error. Some APIs might treat it as a POST request to create a new resource, but this violates REST principles where PUT should be idempotent and target a specific resource.",
                "category": "edge_case",
                "difficulty": "hard",
                "keywords": ["PUT", "HTTP", "REST"]
            },
            {
                "question": "How should an API handle duplicate POST requests?",
                "ground_truth": "APIs should implement idempotency keys for POST requests to handle duplicates. The client sends a unique idempotency key (e.g., UUID) in the request header. The server stores this key and returns the same response for duplicate requests within a time window. This prevents creating duplicate resources due to network retries or client errors.",
                "category": "edge_case",
                "difficulty": "hard",
                "keywords": ["POST", "idempotency", "API design"]
            }
        ]

    def _create_ambiguous_questions(self) -> List[Dict[str, Any]]:
        """Questions that might be ambiguous or require clarification"""
        return [
            {
                "question": "What is the best authentication method?",
                "ground_truth": "There is no single 'best' authentication method as it depends on the use case. JWT is suitable for stateless APIs and microservices. OAuth 2.0 is ideal for third-party integrations. Session-based authentication works well for traditional web applications. Multi-factor authentication (MFA) provides the highest security. The best choice depends on factors like security requirements, scalability needs, and user experience considerations.",
                "category": "ambiguous",
                "difficulty": "hard",
                "keywords": ["authentication", "comparison", "best practices"]
            }
        ]

    def export_dataset(self, dataset: List[Dict[str, Any]], filepath: str):
        """Export dataset to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "domain": self.domain,
                    "total_questions": len(dataset)
                },
                "questions": dataset
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Dataset exported to {filepath}")