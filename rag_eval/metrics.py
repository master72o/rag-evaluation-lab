"""
Aggregate RAG Benchmark Metrics Calculator.
"""

import numpy as np
from typing import List, Dict, Any
from rag_eval.schema import RAGItemEvaluation, RAGConfigType


class RAGMetricsCalculator:
    """Calculates dataset-wide aggregate metrics for RAG pipeline experiments."""

    @staticmethod
    def calculate_config_metrics(evaluations: List[RAGItemEvaluation]) -> Dict[str, Any]:
        if not evaluations:
            return {"total_items": 0, "overall_score": 0.0}

        total = len(evaluations)
        passed = sum(1 for e in evaluations if e.passed)

        p_at_k = [e.retrieval.precision_at_k for e in evaluations]
        r_at_k = [e.retrieval.recall_at_k for e in evaluations]
        mrr_val = [e.retrieval.mrr for e in evaluations]
        ndcg_val = [e.retrieval.ndcg for e in evaluations]

        faith_val = [e.generation.faithfulness for e in evaluations]
        ans_rel = [e.generation.answer_relevance for e in evaluations]
        ctx_prec = [e.generation.context_precision for e in evaluations]
        ctx_rec = [e.generation.context_recall for e in evaluations]
        cit_corr = [e.generation.citation_correctness for e in evaluations]
        scores = [e.overall_score for e in evaluations]

        return {
            "total_items": total,
            "count": total,
            "passed_count": passed,
            "pass_rate": round(passed / total, 4),
            "overall_mean_score": round(float(np.mean(scores)), 4),
            "retrieval": {
                "mean_precision_at_k": round(float(np.mean(p_at_k)), 4),
                "mean_recall_at_k": round(float(np.mean(r_at_k)), 4),
                "mean_mrr": round(float(np.mean(mrr_val)), 4),
                "mean_ndcg": round(float(np.mean(ndcg_val)), 4),
            },
            "generation": {
                "mean_faithfulness": round(float(np.mean(faith_val)), 4),
                "mean_answer_relevance": round(float(np.mean(ans_rel)), 4),
                "mean_context_precision": round(float(np.mean(ctx_prec)), 4),
                "mean_context_recall": round(float(np.mean(ctx_rec)), 4),
                "mean_citation_correctness": round(float(np.mean(cit_corr)), 4),
            }
        }

    @classmethod
    def evaluate_all_configs(
        cls, all_evaluations: Dict[RAGConfigType, List[RAGItemEvaluation]]
    ) -> Dict[str, Any]:
        suite_metrics = {}
        for cfg, evals in all_evaluations.items():
            cfg_key = cfg.value if isinstance(cfg, RAGConfigType) else str(cfg)
            suite_metrics[cfg_key] = cls.calculate_config_metrics(evals)
        return suite_metrics

    calculate_suite_metrics = evaluate_all_configs


MetricsCalculator = RAGMetricsCalculator

