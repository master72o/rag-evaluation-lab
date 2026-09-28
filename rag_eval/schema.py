"""
Data Schemas for RAG Evaluation Lab.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class RAGConfigType(str, Enum):
    V1_BASELINE = "RAG_v1_Baseline"
    V2_HYBRID = "RAG_v2_Hybrid"
    V3_RERANKED = "RAG_v3_Reranked"


@dataclass
class ContextChunk:
    chunk_id: str
    text: str
    score: float = 0.0
    is_relevant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RAGChunk = ContextChunk


@dataclass
class RetrievalMetricsResult:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
        }


@dataclass
class GenerationMetricsResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    citation_correctness: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "citation_correctness": round(self.citation_correctness, 4),
        }


@dataclass
class RAGItemEvaluation:
    item_id: str
    config_name: RAGConfigType
    retrieval: RetrievalMetricsResult
    generation: GenerationMetricsResult
    overall_score: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "config_name": self.config_name.value if isinstance(self.config_name, RAGConfigType) else self.config_name,
            "overall_score": float(self.overall_score),
            "passed": bool(self.passed),
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
        }


@dataclass
class RAGBenchmarkItem:
    id: str
    prompt: str
    reference_answer: str
    ground_truth_chunk_ids: List[str]
    retrieved_chunks_v1: List[ContextChunk]
    retrieved_chunks_v2: List[ContextChunk]
    retrieved_chunks_v3: List[ContextChunk]
    response_v1: str
    response_v2: str
    response_v3: str
    citations_v1: List[str] = field(default_factory=list)
    citations_v2: List[str] = field(default_factory=list)
    citations_v3: List[str] = field(default_factory=list)
    domain: str = "general"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGBenchmarkItem":
        def parse_chunks(raw_list):
            return [ContextChunk(**c) if isinstance(c, dict) else c for c in raw_list]

        return cls(
            id=data["id"],
            prompt=data["prompt"],
            reference_answer=data["reference_answer"],
            ground_truth_chunk_ids=data.get("ground_truth_chunk_ids", []),
            retrieved_chunks_v1=parse_chunks(data.get("retrieved_chunks_v1", [])),
            retrieved_chunks_v2=parse_chunks(data.get("retrieved_chunks_v2", [])),
            retrieved_chunks_v3=parse_chunks(data.get("retrieved_chunks_v3", [])),
            response_v1=data.get("response_v1", ""),
            response_v2=data.get("response_v2", ""),
            response_v3=data.get("response_v3", ""),
            citations_v1=data.get("citations_v1", []),
            citations_v2=data.get("citations_v2", []),
            citations_v3=data.get("citations_v3", []),
            domain=data.get("domain", "general"),
        )


class RAGDataset:
    """Benchmark Dataset loader and batch evaluator."""

    def __init__(self, items: List[RAGBenchmarkItem]):
        self.items = items

    @classmethod
    def from_jsonl(cls, file_path: str) -> "RAGDataset":
        import json
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(RAGBenchmarkItem.from_dict(json.loads(line)))
        return cls(items=items)

    def evaluate_all(self) -> Dict[RAGConfigType, List[RAGItemEvaluation]]:
        from rag_eval.experiment_runner import ExperimentRunner
        results = {
            RAGConfigType.V1_BASELINE: [],
            RAGConfigType.V2_HYBRID: [],
            RAGConfigType.V3_RERANKED: []
        }
        for item in self.items:
            item_evals = ExperimentRunner.evaluate_benchmark_item(item)
            for cfg, ev in item_evals.items():
                results[cfg].append(ev)
        return results

