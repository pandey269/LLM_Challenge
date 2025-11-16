"""Evaluation harness covering README test cases."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from .graph import run_graph
from .models import EvaluationResult

SAMPLE_QUERIES_PATH = Path("sample_data/sample_queries.json")


def run_batch_evaluation() -> list[EvaluationResult]:
    with open(SAMPLE_QUERIES_PATH, encoding="utf-8") as handle:
        dataset = json.load(handle)
    results: list[EvaluationResult] = []
    latencies: list[float] = []
    for item in dataset:
        query_id = item["id"]
        question = item["question"]
        expected = item.get("expected_answer")
        start = perf_counter()
        answer = run_graph(question)
        latency = (perf_counter() - start) * 1000
        latencies.append(latency)
        ragas_scores = evaluate(
            model_outputs=[{"question": question, "answer": answer.answer, "contexts": answer.evidence}],
            metrics=[answer_relevancy, faithfulness],
        )
        metrics_df = ragas_scores.to_pandas()
        result = EvaluationResult(
            query_id=query_id,
            exact_match=1.0 if expected and expected.lower() in answer.answer.lower() else 0.0,
            semantic_similarity=float(metrics_df["answer_relevancy"].iloc[0]),
            grounding_score=float(metrics_df["faithfulness"].iloc[0]),
            latency_stats={"latency_ms": latency},
            token_usage={"prompt": 0, "completion": 0},
        )
        results.append(result)
    summary = {
        "p50": sorted(latencies)[len(latencies) // 2],
        "p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
    }
    output_path = Path("data/evaluations/latest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return results
