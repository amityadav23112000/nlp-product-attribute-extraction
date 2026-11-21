# Transformer validation tuning summary

All five completed runs passed identical validation ID/token/gold checks. Independent exact-span metrics reproduce saved seqeval scores; best-epoch and completion records agree with learning histories.

| Learning rate | Seed | Best epoch | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1e-05 | 42 | 15 | 70.08% | 54.52% |
| 2e-05 | 42 | 13 | 78.16% | 68.34% |
| 5e-05 | 42 | 11 | 79.94% | 77.64% |
| 5e-05 | 43 | 14 | 81.12% | 78.08% |
| 5e-05 | 44 | 15 | 80.74% | 78.30% |

Selected learning rate: **5e-05** (seed-42 validation selection).

Three-seed micro F1 mean: **80.60%**; sample standard deviation: **0.60 percentage points**.

Maximum 15 epochs, early-stopping patience 3, batch size 16, weight decay 0.01. Each run uses its validation-selected best checkpoint. Final training on train + validation is still pending.

The official test set was not evaluated. These scores describe the preserved processed BIO benchmark with gold-assisted token boundaries; raw-text inference quality requires separate measurement.

Model revision: `not pinned in historical study`.

Model weights are stored in scratch; configurations, histories, metrics and validation predictions are retained with the project.
