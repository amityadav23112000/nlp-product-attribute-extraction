# BERT validation diagnosis — 9 October 2026

## Evaluation audit

The shared evaluator confirms identical validation IDs, product IDs, tokens, gold labels, sources and categories across CRF and archived BERT predictions. Training and validation products/clusters are disjoint. Metrics computed by exact attribute/start/end matching reproduce archived BERT seqeval precision, recall, micro/macro F1 and token accuracy to numerical precision.

The evaluator retains the historical permissive interpretation of malformed predicted I-tags as entity starts. Exact entity spans remain the primary metric; switching to strict BIO rejection would change the protocol and must be reported as a separate ablation.

Six regression checks exercise missing-ID/gold mismatch guards, boundary and attribute errors, malformed I-tags and one-to-one error matching. The error taxonomy reconstructs exact false-positive and false-negative totals for all compared models.

## Initial five-epoch result

The archived rerun scored 0.630588 micro F1 versus CRF's 0.789439. Its errors comprise 152 complete misses, 360 spurious predictions, 168 boundary errors and 204 attribute confusions. CRF has 266 misses, 104 spurious predictions, 76 boundary errors and 58 confusions.

Thus BERT's loss is not simply failure to find entities: precision, boundaries and attribute assignment are substantial problems. Its tail-group micro F1 is 0.2564 versus CRF's 0.7458, and its macro F1 is 0.4047 versus 0.7596. Both archived BERT runs have over 140 illegal I transitions. These are diagnostics, not evidence of a confirmed trainer defect.

Code inspection confirms that special tokens and subsequent subwords are masked with -100, padding is handled by the token-classification collator, predictions are mapped back using the loss mask, and length/word coverage checks protect against dropped words. All recorded training runs pass finite-loss checks and show falling loss. A training-only memorization probe is therefore deferred unless further diagnostics identify a suspected implementation defect.

## Bounded tuning evidence

The recorded study tests learning rates 1e-5, 2e-5 and 5e-5 at batch size 16, maximum 15 epochs, weight decay 0.01 and seed 42. Early stopping uses three consecutive non-improving epochs. The selected learning rate is repeated at predetermined seeds 43 and 44; confirmation scores do not change the learning-rate selection.

| Learning rate | Best epoch | Validation micro F1 |
| --- | ---: | ---: |
| 1e-5 | 15 | 0.700769 |
| 2e-5 | 13 | 0.781609 |
| 5e-5 | 11 | 0.799389 |

The longer training horizon also changes the linear learning-rate decay schedule, so this study does not isolate epoch count as the sole cause of improvement. It nevertheless shows that the initial five-epoch setup was not an adequate estimate of this model's achievable validation performance.

For the selected seed-42 run, macro F1 is 0.7764. Tail-group F1 rises to 0.7419, close to CRF's 0.7458. Invalid I transitions fall to 59. There are still more boundary errors (110 versus 76) and spurious predictions (196 versus 104), but fewer misses (113 versus 266).

Source/category evidence is mixed: title F1 rises to 0.8243 versus 0.8014, while description F1 is 0.7784 versus 0.7789. Grocery F1 rises to 0.6575 versus 0.5739. Do not claim improvement across every source/category or generalization to the official test set.

Detailed per-attribute metrics, examples, training span frequencies and all error counts are in `validation_comparison/comparison.json`. Selected settings, seed results, training histories and checkpoints are recorded under `artifacts/bert_lr_study_20261009`; large weights reside under `/scratch/csisnlp_09/models/bert_lr_study_20261009`.

## Raw-text inference limitation

The preserved benchmark forces gold annotation boundaries into word tokenization. `src/tokenization.py` provides annotation-independent regex tokens and exact character offsets without those boundaries. Four regression checks cover source spacing, Unicode offsets, absent gold-based splitting and invalid spans. This is an inference foundation, not a complete deployed extractor or an alternative benchmark score.

## Next gate

The predetermined seed study completed: seeds 42/43/44 scored 0.799389/0.811223/0.807362, with mean 0.805991 and sample standard deviation 0.006035. All five study runs pass the shared validation-data checks and independently reproduce their saved metrics. Best-epoch and completion records are consistent with their learning histories. The selected settings are preserved in `configs/bert_selected.json`.

Next establish DeBERTa on the same validation protocol. Freeze final retraining rules and frequency thresholds before official test evaluation. No official test predictions were loaded during this work.
