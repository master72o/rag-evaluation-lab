"""
Tests for Metrics module and CLI.
"""

from rag_eval.schema import RAGDataset, RAGConfigType
from rag_eval.metrics import MetricsCalculator


def test_metrics_calculator(tmp_path):
    jsonl_content = '{"id":"rag_001","domain":"tech","prompt":"P?","reference_answer":"A.","ground_truth_chunk_ids":["c1"],"retrieved_chunks_v1":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v1":"R1","citations_v1":["c1"],"retrieved_chunks_v2":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v2":"R2","citations_v2":["c1"],"retrieved_chunks_v3":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v3":"R3","citations_v3":["c1"]}\n'
    file_path = tmp_path / "test.jsonl"
    file_path.write_text(jsonl_content)

    dataset = RAGDataset.from_jsonl(str(file_path))
    evaluations = dataset.evaluate_all()
    suite_metrics = MetricsCalculator.calculate_suite_metrics(evaluations)

    assert "RAG_v1_Baseline" in suite_metrics
    assert suite_metrics["RAG_v1_Baseline"]["count"] == 1
    assert "overall_mean_score" in suite_metrics["RAG_v1_Baseline"]
