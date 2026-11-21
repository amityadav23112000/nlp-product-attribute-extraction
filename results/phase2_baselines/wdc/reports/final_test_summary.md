# WDC-PAVE final official test results

CRF historical official test micro F1: 79.34%. Transformers retrained on train plus validation; final epoch only.

| Model | Seed | Precision | Recall | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bert | 42 | 81.39% | 85.20% | 83.25% | 81.08% |
| bert | 43 | 80.97% | 85.02% | 82.95% | 79.72% |
| bert | 44 | 81.48% | 85.11% | 83.26% | 80.15% |
| deberta | 42 | 84.03% | 87.13% | 85.55% | 83.48% |
| deberta | 43 | 84.83% | 87.68% | 86.23% | 82.08% |
| deberta | 44 | 84.21% | 86.89% | 85.53% | 82.04% |

bert mean: 83.15%; sample standard deviation: 0.18 percentage points.


deberta mean: 85.77%; sample standard deviation: 0.40 percentage points.

Benchmark uses gold-assisted token boundaries. All predetermined seeds are reported; no test-based model selection.
