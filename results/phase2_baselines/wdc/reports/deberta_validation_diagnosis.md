# DeBERTa development evaluation — 9 October 2026

## Model and environment

The handover's candidate is `microsoft/deberta-v3-base`, pinned to revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Microsoft documents this checkpoint in its [model card](https://huggingface.co/microsoft/deberta-v3-base); the [Transformers DeBERTa-v2 documentation](https://huggingface.co/docs/transformers/main/model_doc/deberta-v2) describes the SentencePiece implementation used by the loaded model/tokenizer classes.

Actual runtime classes are `DebertaV2ForTokenClassification` and `DebertaV2TokenizerFast`. The model has 183,910,759 parameters including the new 103-label classifier. Runtime versions: Python 3.11.17, torch 2.14.0+cu130, Transformers 4.57.6, SentencePiece 0.2.2 and protobuf 7.36.2. BF16 training ran on the RTX PRO 6000 GPU through course SLURM batch job 386761. The previous interactive allocation ended before submission; only one new batch job was submitted.

## Tokenizer checks

The development-only audit passed for 1,812 training and 320 validation sequences. Maximum subword lengths are 209 and 188 respectively, with no dropped words, label-length errors, unknown labels or invalid gold BIO transitions. The model's positional limit was also checked before training. First-subword-only supervision and word-level prediction export remain unchanged.

The fast tokenizer converter emits a byte-fallback warning. A separate native/fast SentencePiece comparison on all 2,132 development sequences found identical token IDs and zero unknown tokens in both implementations. This establishes compatibility on the current development data; it does not guarantee arbitrary unseen Unicode input behaves identically.

Transformers also emits a Mistral regex warning in offline mode. Inspection of the installed `tokenization_utils_base.py` shows that this warning path can be triggered for large-vocabulary non-Mistral tokenizers when checkpoint config version metadata is absent. The warning branch does not replace the pre-tokenizer. No Mistral regex patch was applied to DeBERTa; native/fast agreement provides direct evidence that current development input was not changed by that warning.

The complete audits and preparation manifest are under `artifacts/deberta_study_20261009_154943_386761_preparation/`.

## Controlled learning-rate search

The same three learning rates, maximum 15 epochs, patience 3, batch size 16, weight decay 0.01 and seed-42 selection policy used for BERT were applied to DeBERTa. The pretrained model and tokenizer were loaded from the pinned local cache during training. Special/subsequent subword labels are masked with -100, and no gold entities are silently truncated.

| Learning rate | Best epoch | Validation micro F1 |
| --- | ---: | ---: |
| 1e-5 | 13 | 0.744674 |
| 2e-5 | 13 | 0.823888 |
| 5e-5 | 15 | 0.848392 |

Learning rate 5e-5 was selected solely from seed-42 validation scores, then repeated with predetermined seeds 43 and 44. Scores were 0.848392, 0.844431 and 0.840403: mean 0.844409, sample standard deviation 0.003995. The mean is 3.84 percentage points above tuned BERT's three-seed mean and 5.50 points above the recorded CRF validation result. These comparisons are development evidence, not a statistical-significance or official test claim.

The study verifier checked all five runs' training/validation dataset hashes, identical validation prediction inputs, saved-versus-independent metrics, best epochs, completed histories and seed-42 learning-rate selection. The three-seed mean and sample standard deviation were independently recomputed. Full results are in `deberta_tuning_summary.md`; per-attribute/source/category/error/frequency-group results are in `deberta_validation_comparison/` and `deberta_seed_comparison/`.

## Selected seed-42 comparison

| Metric | CRF | Tuned BERT | DeBERTa |
| --- | ---: | ---: | ---: |
| Micro F1 | 0.7894 | 0.7994 | 0.8484 |
| Macro F1 | 0.7596 | 0.7764 | 0.8235 |
| Title F1 | 0.8014 | 0.8243 | 0.8736 |
| Description F1 | 0.7789 | 0.7784 | 0.8271 |
| Tail-group F1 | 0.7458 | 0.7419 | 0.8358 |
| Office Products F1 | 0.6628 | 0.6658 | 0.8000 |

DeBERTa has 81 complete misses, 154 spurious predictions, 85 boundary errors and 45 attribute confusions in this comparison. Tuned BERT has 113, 196, 110 and 64 respectively. CRF retains fewer spurious predictions and boundary errors than DeBERTa. The error taxonomy uses one-to-one matching and must reconstruct exact false-positive/false-negative totals.

Gains are not universal: seed-42 DeBERTa Grocery F1 is 0.6438 versus BERT's 0.6575, and Computers F1 is 0.9127 versus 0.9132. Tail supports are small; an extreme-tail group with zero validation support should not be interpreted as a measured extraction failure.

## Interpretation and next step

These are development results on the preserved BIO benchmark, with exact attribute/span matching and the historical permissive treatment of illegal predicted I-tags. Gold-assisted word token boundaries remain a benchmark limitation. No official test predictions were loaded, and no raw-text deployment claim follows from these scores.

After seed confirmation and verification, freeze BERT and DeBERTa hyperparameters and the final epoch/retraining policy before combining the 906 training and 160 validation products. Official test evaluation is the next project phase. Normalization and end-to-end raw-input inference remain separate subsequent work.
