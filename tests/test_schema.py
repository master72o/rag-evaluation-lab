"""
Tests for Schema module in RAG Evaluation Lab.
"""

from rag_eval.schema import RAGChunk, RAGBenchmarkItem, RAGDataset, RAGConfigType


def test_rag_chunk_creation():
    chunk = RAGChunk(chunk_id="chk_01", text="Sample chunk text", score=0.95)
    assert chunk.chunk_id == "chk_01"
    assert chunk.score == 0.95


def test_rag_benchmark_item_parsing():
    data = {
        "id": "rag_001",
        "domain": "finance",
        "prompt": "Test prompt?",
        "reference_answer": "Test answer.",
        "ground_truth_chunk_ids": ["chk_01"],
        "retrieved_chunks_v1": [{"chunk_id": "chk_01", "text": "Sample text", "score": 0.9}],
        "response_v1": "Test response",
        "citations_v1": ["chk_01"],
        "retrieved_chunks_v2": [{"chunk_id": "chk_01", "text": "Sample text", "score": 0.9}],
        "response_v2": "Test response",
        "citations_v2": ["chk_01"],
        "retrieved_chunks_v3": [{"chunk_id": "chk_01", "text": "Sample text", "score": 0.9}],
        "response_v3": "Test response",
        "citations_v3": ["chk_01"]
    }
    item = RAGBenchmarkItem.from_dict(data)
    assert item.id == "rag_001"
    assert item.domain == "finance"
    assert len(item.retrieved_chunks_v1) == 1
    assert item.retrieved_chunks_v1[0].chunk_id == "chk_01"


def test_rag_dataset_load(tmp_path):
    jsonl_content = '{"id":"rag_001","domain":"tech","prompt":"P?","reference_answer":"A.","ground_truth_chunk_ids":["c1"],"retrieved_chunks_v1":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v1":"R1","citations_v1":["c1"],"retrieved_chunks_v2":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v2":"R2","citations_v2":["c1"],"retrieved_chunks_v3":[{"chunk_id":"c1","text":"txt","score":0.9}],"response_v3":"R3","citations_v3":["c1"]}\n'
    file_path = tmp_path / "test.jsonl"
    file_path.write_text(jsonl_content)

    dataset = RAGDataset.from_jsonl(str(file_path))
    assert len(dataset.items) == 1
    evals = dataset.evaluate_all()
    assert RAGConfigType.V1_BASELINE in evals
    assert len(evals[RAGConfigType.V1_BASELINE]) == 1
