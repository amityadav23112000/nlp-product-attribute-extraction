# CPEC committee - MAVE

Voters per seed: crf, deberta (CRF shared across seeds).
Thresholds selected on validation only: `{"head": 1, "medium": 1, "tail": 1, "extreme_tail": 1}`

| Split | System | Seed 42 | Seed 43 | Seed 44 | Mean micro-F1 | SD (pp) | Mean P | Mean R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | DeBERTa | 82.85 | 82.93 | 82.08 | **82.62** | 0.47 | 78.01 | 87.84 |
| validation | CRF | 80.53 | - | - | **80.53** | - | 80.13 | 80.92 |
| validation | Committee_majority | 80.47 | 80.80 | 80.85 | **80.71** | 0.21 | 85.14 | 76.72 |
| validation | CPEC_committee | 83.13 | 82.91 | 82.26 | **82.76** | 0.45 | 75.98 | 90.89 |
| test | DeBERTa | 83.61 | 83.85 | 84.47 | **83.97** | 0.44 | 80.17 | 88.16 |
| test | CRF | 80.81 | - | - | **80.81** | - | 81.23 | 80.40 |
| test | Committee_majority | 80.85 | 81.09 | 81.60 | **81.18** | 0.39 | 86.27 | 76.65 |
| test | CPEC_committee | 83.67 | 83.54 | 83.78 | **83.66** | 0.12 | 77.77 | 90.52 |

## Test error profile (seed 42, span level)

| System | correct | spurious | boundary | wrong attribute | missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeBERTa | 1382 | 280 | 33 | 29 | 144 |
| CPEC_committee | 1427 | 348 | 25 | 29 | 104 |

## Test micro-F1 by attribute-frequency band (seed 42)

| Band | support | DeBERTa | CPEC |
| --- | ---: | ---: | ---: |
| head | 1448 | 84.02 | 84.14 |
| medium | 130 | 79.85 | 79.27 |
| tail | 2 | 66.67 | 66.67 |
| extreme_tail | 2 | 0.00 | 0.00 |

Entity-level strict span F1 (seqeval), identical to the Phase 2 metric. No test-based selection.
