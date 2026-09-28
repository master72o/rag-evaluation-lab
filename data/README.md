# RAG Benchmark Dataset Documentation

This directory contains the benchmark dataset used to evaluate RAG retrieval and generation capabilities across pipeline configurations.

## Schema Specification

Each line in `rag_benchmark_dataset.jsonl` is a JSON object formatted as follows:

```json
{
  "id": "rag_001",
  "domain": "finance",
  "prompt": "What were Acme Corp's net revenues and operating expenses in Q3 2025?",
  "reference_answer": "In Q3 2025, Acme Corp reported net revenues of $4.2B and operating expenses of $1.8B.",
  "ground_truth_chunk_ids": ["chk_fin_101", "chk_fin_102"],
  "retrieved_chunks_v1": [
    {"chunk_id": "chk_fin_105", "text": "Acme Corp Q3 overview...", "score": 0.72},
    {"chunk_id": "chk_fin_101", "text": "Acme Corp reported net revenues of $4.2B in Q3 2025.", "score": 0.68}
  ],
  "response_v1": "Acme Corp reported revenues of $4.2B.",
  "citations_v1": ["chk_fin_101"],
  "retrieved_chunks_v2": [
    {"chunk_id": "chk_fin_101", "text": "Acme Corp reported net revenues of $4.2B in Q3 2025.", "score": 0.85},
    {"chunk_id": "chk_fin_102", "text": "Operating expenses for Q3 2025 totaled $1.8B.", "score": 0.81}
  ],
  "response_v2": "In Q3 2025, Acme Corp had net revenues of $4.2B and operating expenses of $1.8B.",
  "citations_v2": ["chk_fin_101", "chk_fin_102"],
  "retrieved_chunks_v3": [
    {"chunk_id": "chk_fin_101", "text": "Acme Corp reported net revenues of $4.2B in Q3 2025.", "score": 0.96},
    {"chunk_id": "chk_fin_102", "text": "Operating expenses for Q3 2025 totaled $1.8B.", "score": 0.94}
  ],
  "response_v3": "In Q3 2025, Acme Corp reported net revenues of $4.2B and operating expenses of $1.8B.",
  "citations_v3": ["chk_fin_101", "chk_fin_102"]
}
```

## Data Quality & Coverage
- Domain diversity: Finance, Healthcare, Legal, Technical Documentation.
- Synthetic ground-truth mapping created via human expert verification.
