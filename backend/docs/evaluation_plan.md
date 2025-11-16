# Evaluation Plan

| ID | README Test | Approach |
|----|-------------|----------|
| A1 | Single-Document Fact Retrieval | Ingest `ml_basics.pdf`; query via CLI/API and verify citations referencing same doc/page. |
| A2 | Multi-Document Synthesis | Ingest `climate_change.pdf` + `renewable_energy.pdf`; query Q2 and ensure citations from both doc IDs. |
| A3 | Out-of-Corpus Handling | With only biology docs loaded, query "What is dark matter?" and assert `not_found_reason` is populated. |
| A4 | Conflicting Evidence | Provide docs with differing statistics; system must surface both values and cite both chunk IDs. |
| A5 | Long-Context Robustness | Use ≥80-page PDF; confirm retrieved chunk metadata includes requested section/page. |
| B1 | Spanish Query | Ingest Spanish doc, run Spanish question, ensure response + citations in Spanish. |
| B2 | Cross-Lingual Query | Ingest English doc, ask question in Hindi; response uses Hindi while citing English documents. |
| C1 | Index Scale Test | Generate ≥5,000 chunks (seeded via synthetic docs), collect latency metrics from `evaluation.py`. Goal: p95 < 2.5s. |
| C2 | Concurrency Test | Use Locust/K6 hitting `/query` with 20 concurrent users for 2 min; ensure no errors, stable latency logged. |
| C3 | Cold vs Warm Cache | Compare metrics from CLI evaluation with caching disabled/enabled; record in `data/evaluations/cache_comparison.json`. |

All evaluation scripts log detailed outputs into `data/evaluations/` for inclusion in the final report.
