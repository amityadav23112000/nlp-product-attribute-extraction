# microsoft/deberta-v3-base validation tuning summary

All five completed runs passed identical validation ID/token/gold checks. Independent exact-span metrics reproduce saved seqeval scores; best-epoch and completion records agree with learning histories.

| Learning rate | Seed | Best epoch | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1e-05 | 42 | 13 | 74.47% | 58.29% |
| 2e-05 | 42 | 13 | 82.39% | 73.68% |
| 5e-05 | 42 | 15 | 84.84% | 82.35% |
| 5e-05 | 43 | 12 | 84.44% | 82.75% |
| 5e-05 | 44 | 15 | 84.04% | 82.47% |

Selected learning rate: **5e-05** (seed-42 validation selection).

Three-seed micro F1 mean: **84.44%**; sample standard deviation: **0.40 percentage points**.

Maximum 15 epochs, early-stopping patience 3, batch size 16, weight decay 0.01. Each run uses its validation-selected best checkpoint. Final training on train + validation is still pending.

The official test set was not evaluated. These scores describe the preserved processed BIO benchmark with gold-assisted token boundaries; raw-text inference quality requires separate measurement.

Model revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.

Model weights are stored in scratch; configurations, histories, metrics and validation predictions are retained with the project.
