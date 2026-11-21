# CPEC committee - WDC

Voters per seed: crf, bert, deberta (CRF shared across seeds).
Thresholds selected on validation only: `{"head": 2, "medium": 2, "tail": 2, "extreme_tail": 3}`

| Split | System | Seed 42 | Seed 43 | Seed 44 | Mean micro-F1 | SD (pp) | Mean P | Mean R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | DeBERTa | 84.84 | 84.44 | 84.04 | **84.44** | 0.40 | 82.55 | 86.42 |
| validation | CRF | 78.94 | - | - | **78.94** | - | 83.40 | 74.94 |
| validation | BERT | 79.94 | 81.12 | 80.74 | **80.60** | 0.60 | 78.69 | 82.60 |
| validation | Committee_majority | 84.39 | 85.39 | 85.14 | **84.98** | 0.52 | 87.54 | 82.56 |
| validation | CPEC_committee | 84.39 | 85.39 | 85.14 | **84.98** | 0.52 | 87.54 | 82.56 |
| test | DeBERTa | 85.55 | 86.23 | 85.53 | **85.77** | 0.40 | 84.36 | 87.23 |
| test | CRF | 79.34 | - | - | **79.34** | - | 84.08 | 75.10 |
| test | BERT | 83.25 | 82.95 | 83.26 | **83.15** | 0.18 | 81.28 | 85.11 |
| test | Committee_majority | 87.04 | 87.05 | 86.89 | **86.99** | 0.09 | 89.30 | 84.81 |
| test | CPEC_committee | 87.04 | 87.05 | 86.89 | **86.99** | 0.09 | 89.30 | 84.81 |

## Test error profile (seed 42, span level)

| System | correct | spurious | boundary | wrong attribute | missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeBERTa | 2978 | 212 | 199 | 155 | 147 |
| CPEC_committee | 2898 | 135 | 114 | 94 | 321 |

## Test micro-F1 by attribute-frequency band (seed 42)

| Band | support | DeBERTa | CPEC |
| --- | ---: | ---: | ---: |
| head | 2910 | 86.16 | 87.50 |
| medium | 448 | 83.35 | 85.42 |
| tail | 60 | 73.02 | 76.11 |

Entity-level strict span F1 (seqeval), identical to the Phase 2 metric. No test-based selection.
