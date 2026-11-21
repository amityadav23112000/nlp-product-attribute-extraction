# Work completed: product attribute extraction and normalization

WDC final evaluation completion (10 October 2026): BERT three-seed official test micro F1 mean 83.15%, sample standard deviation 0.18 percentage points; DeBERTa mean 85.77%, sample standard deviation 0.40 percentage points. Historical CRF official test micro F1 remains 79.34%. Transformers retrained on 1,066 train-plus-validation products with frozen budgets of 14 BERT epochs and 15 DeBERTa epochs, then evaluated all predetermined seeds on 354 official test products. Independent saved prediction, metric, fixed-history and aggregate checks passed. See the [WDC final report](../artifacts/wdc_final_20261010_085407_387830/summary.md). Earlier pending-status statements below are historical; both planned dataset experiments are now complete under their documented protocols. Normalization and the raw-text demonstration remain pending; full MAVE benchmarking would be separate additional work.

Final MAVE evaluation update: CRF and all three fixed-budget DeBERTa models completed training on 4,284 train-plus-validation products, followed by evaluation on the 716-product custom pilot test. CRF test micro F1: 80.81%; DeBERTa seeds 42/43/44: 83.61%/83.85%/84.47%; mean 83.97%, sample standard deviation 0.44 percentage points. All DeBERTa models used exactly 10 epochs. Independent prediction, gold, hash, metric, history and aggregate checks passed. See the [final test report](../artifacts/mave_final_20261009_210933_387301/summary.md). Statements below describing pending MAVE testing refer to earlier snapshots. WDC Transformer final evaluation, normalization and raw-text inference remain outstanding.

Latest update: MAVE DeBERTa confirmation seeds 43 and 44 completed successfully. Validation micro F1 for seeds 42/43/44 is 82.85%/82.93%/82.08%; mean 82.62%, sample standard deviation 0.47 percentage points. All runs outperform the CRF baseline (80.53%) on identical frozen validation inputs. Provenance, configuration, shared evaluation and summary checks passed. Test evaluation remains pending. See the [three-seed summary](../results/mave_confirmation_387172/summary.md). Next freeze final training/evaluation rules before held-out testing.

Updated: 9 October 2026. This report describes verified work and saved results through the completed MAVE CRF baseline. Repository: `/home/csisnlp_09/nlp_project`.

Subsequent completion: the MAVE DeBERTa baseline also completed successfully (job 387147, 34m51s). Best epoch 10, early stopping at epoch 13; validation precision 79.48%, recall 86.52%, micro F1 82.85%, macro F1 70.87%. Micro F1 exceeds CRF by 2.33 percentage points on identical validation inputs. Prediction/gold identity, split separation, error reconciliation and learning-history consistency checks passed. This is a single seed-42 development result; the test split remains untouched. See the [CRF/DeBERTa comparison](../results/mave_deberta_20261009_184522_387147/comparison.md). Earlier descriptions of pending baseline training below refer to the report's original snapshot; next work is freezing final evaluation rules or further development experiments as justified.

## Project objective and current status

The project extracts attribute-value spans from product text, compares a CRF with Transformer models, and studies performance across datasets and attribute frequencies. Attribute-aware value normalization and a raw-text inference demonstration remain to be implemented and evaluated.

WDC-PAVE preprocessing, the CRF baseline, and BERT/DeBERTa development studies are complete. MAVE source acquisition, controlled subset reconstruction, data audits, and its first CRF development experiment are also complete. MAVE Transformer training and final Transformer test evaluations are pending. This document does not imply that a new training job has been launched.

## 1. WDC-PAVE data and baseline

The fixed WDC-PAVE corpus contains 1,420 products: 906 training, 160 validation, and 354 official test products. Its predefined benchmark inventory contains 51 attributes and 103 BIO labels. Processed data is stored at `/scratch/csisnlp_09/data/wdc_pave/processed/crf`.

Completed work includes conversion to token-level BIO records, data checks, CRF feature preparation, training, prediction export, and entity-level evaluation. The CRF uses lexical, shape, neighboring-token, and source features.

| WDC-PAVE CRF evaluation | Entity micro F1 |
| --- | ---: |
| Validation | 78.94% |
| Official test | 79.34% |

These are distinct evaluations on their respective splits. The completed CRF test result is preserved; BERT and DeBERTa official test evaluations have not yet been performed.

## 2. Shared evaluation and experiment auditing

`scripts/compare_validation.py` provides a common evaluation path. It checks that predictions match the reference validation IDs, tokens, gold labels, and metadata, and checks train/validation product and cluster separation. It reports precision, recall, micro/macro entity F1, token accuracy, attribute/source/category breakdowns, and training-frequency groups.

The one-to-one error analysis separates complete misses, spurious predictions, boundary errors, and attribute confusions. Reconstructed false-positive and false-negative totals are reconciled with exact entity matching. Independent scores are cross-checked against the historical `seqeval` interpretation of BIO tags, including its treatment of invalid I-tags; alternative decoding would require a separate experiment.

Saved experiment records include configurations, dataset hashes, software versions, learning histories, best checkpoints, metrics, and validation predictions. The study summarizer verifies these records before generating the model-selection summaries.

## 3. BERT development study on WDC-PAVE

GPU execution, first-subword supervision, WordPiece alignment, prediction export, and tokenizer checks were implemented and verified. An initial five-epoch rerun achieved 63.06% validation micro F1. A bounded development study then compared learning rates 1e-5, 2e-5, and 5e-5, allowing up to 15 epochs with early-stopping patience 3, batch size 16, and weight decay 0.01.

The seed-42 validation comparison selected learning rate 5e-5. The selected configuration was repeated with seeds 43 and 44.

| Selected BERT run | Best epoch | Validation micro F1 |
| --- | ---: | ---: |
| Seed 42 | 11 | 79.94% |
| Seed 43 | 14 | 81.12% |
| Seed 44 | 15 | 80.74% |
| Three-seed mean | — | **80.60%** |

Sample standard deviation: **0.60 percentage points**. Each development score uses that run's validation-selected best checkpoint. The historical BERT study did not pin its model revision; this provenance limitation is recorded in the verified summary.

Evidence: [verified BERT summary](../results/bert_tuning_verified_summary.md), [selected configuration](../configs/bert_selected.json), and `artifacts/bert_lr_study_20261009/`.

## 4. DeBERTa development study on WDC-PAVE

DeBERTa-v3-base was integrated with the audited training/evaluation pipeline. Its own tokenizer alignment was checked, and native/fast tokenizers produced identical input IDs on the audited development inputs without unknown tokens. The model revision is pinned to `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.

The same three-learning-rate development search selected 5e-5, followed by confirmation with seeds 43 and 44. The maximum epoch budget, patience, batch size, and weight decay matched the BERT study.

| Selected DeBERTa run | Best epoch | Validation micro F1 |
| --- | ---: | ---: |
| Seed 42 | 15 | 84.84% |
| Seed 43 | 12 | 84.44% |
| Seed 44 | 15 | 84.04% |
| Three-seed mean | — | **84.44%** |

Sample standard deviation: **0.40 percentage points**. All five development runs passed common dataset/metric verification.

Evidence: [DeBERTa summary](../results/deberta_tuning_summary.md), [selected configuration](../configs/deberta_selected.json), and `artifacts/deberta_study_20261009_154943_386761/`.

## 5. MAVE acquisition and reconstruction

The official MAVE release supplies annotations that must be joined with Amazon Review Data (2018) product metadata to recover the text. The user supplied an authorized metadata download URL. Positive and negative label archives and the full metadata archive were downloaded, and their SHA-256 checksums were recorded.

The metadata archive is 12,237,944,088 bytes, approximately 11.4 GiB. Label indexing identified 2,226,509 positive products and 1,248,009 negative products; these counts must not be added as a count of unique products because the inventories can overlap.

The reconstruction pipeline uses the pinned official MAVE cleaner and preserves paragraph order and source fields. It requires evidence values and character offsets to match cleaned text exactly. Positive and negative annotations are merged by product identity; contradictions, unrepresentable spans, duplicate rows, and flat-BIO overlap decisions are explicitly accounted for.

The first reconstruction attempt failed because the metadata archive has two gzip layers. The reader was corrected to decode nested gzip, reject non-JSON payloads early, and verify both gzip CRCs. Regression tests cover this repair. The successful reconstruction job was `387113`; the failed attempt and its report were retained separately.

The successful scan read 15,023,059 metadata rows and matched all 15,000 selected candidate product IDs. Duplicate metadata rows were counted separately. There were 14,725 valid candidate products after exclusions: 267 overlength products, seven products with dropped tokenizer words, and one evidence-offset mismatch were excluded.

Evidence: [MAVE protocol and provenance](MAVE.md), [release manifest](../results/mave_release_manifest.json), [metadata manifest](../results/mave_metadata_manifest.json), and [independent audit](../results/mave_independent_audit.json).

## 6. Frozen MAVE subset

The user confirmed continuing with the subset approach. The prepared dataset is a custom 5,000-product pilot, with 1,000 products each from Shoes, Shirts & Tops, Watches, Books, and Dresses. It is not the full MAVE dataset or a reproduction of the publisher's original benchmark splits.

Candidates and final selections use deterministic seeded hash rankings. Exact normalized text duplicates and available parent-ASIN families are grouped transitively before stable group-level splitting. Whole products exceeding either model's 512-subword limit are excluded rather than truncated.

| Split | Products | Paragraph sequences | Tokens |
| --- | ---: | ---: | ---: |
| Training | 3,548 | 43,302 | 743,574 |
| Validation | 736 | 8,789 | 150,642 |
| Test | 716 | 9,027 | 151,950 |
| Total | **5,000** | **61,118** | **1,046,166** |

The public predefined inventory contains 28 attributes and 57 BIO labels. Positive training examples cover 25 attributes; structural inspection found two test-only positive BABY_DEPARTMENT spans. This is a data-support observation, not a model test evaluation. The frozen splits were not altered in response to this observation.

Flat BIO conversion discarded 620 overlapping evidence spans and retained an overlap log. BIO validity, label membership, word coverage, maximum sequence length, split hashes, and product/group/exact-text separation checks passed. Nineteen MAVE preparation tests and ten existing metric/tokenization tests passed during preparation verification.

Prepared data: `/scratch/csisnlp_09/data/mave/processed/pilot_v2/`. Small immutable preparation records are retained in `artifacts/mave_pilot_v2/`. All MAVE model comparisons should use these same frozen files and the predefined schema.

## 7. Completed MAVE CRF baseline

SLURM job `387136` completed with exit code 0. The baseline used the existing WDC CRF features, L-BFGS, c1=0.1, c2=0.1, all possible transitions, and 150 iterations. It trained only on the training split and evaluated only validation. Allocation elapsed time was 7m42s, with approximately 2.1 GiB peak memory.

| Validation metric | Score |
| --- | ---: |
| Precision | 80.13% |
| Recall | 80.92% |
| Micro entity F1 | **80.53%** |
| Macro entity F1 | 68.80% |

Frozen dataset checks, train/validation separation, independent `seqeval` agreement, and error-count reconstruction passed. Error analysis found 242 complete misses, 258 spurious predictions, 47 attribute confusions, and 21 boundary errors. Source F1 was 89.69% for title, 75.73% for description, and 70.13% for feature. Rare-attribute and zero-support scores require their support counts when interpreted.

This is one fixed CRF development run, without a CRF hyperparameter search or multi-seed study. MAVE scores and WDC scores describe different corpora and are not a direct model-family comparison across datasets.

Evidence: [validation comparison and error analysis](../results/mave_crf_20261009_182845_387136/comparison.md), [exact metrics](../artifacts/mave_crf_20261009_182845_387136/validation_metrics.json), and [completion record](../artifacts/mave_crf_20261009_182845_387136/completed.json).

Reproduce submission from the project directory with `sbatch scripts/mave_crf.sbatch`. The launcher creates new timestamped run directories and requires the prepared scratch dataset and working environment. Training belongs on an allocated compute node.

## 8. Interpretation and remaining work

The current processed BIO benchmarks use gold-assisted token boundaries. An annotation-independent tokenizer foundation and its checks exist, but the complete raw-text inference path and its evaluation are pending. These benchmark scores therefore do not establish raw-input deployment performance.

MAVE's absent attribute annotations do not prove that an attribute is absent. The pilot evaluates released observed evidence after flat-BIO conversion. Category selection, whole-product length filtering, discarded overlaps, and possible near duplicates limit generalization. No equivalence between WDC and MAVE attribute inventories has been assumed.

Next steps, in the user's selected order:

1. Train and validate the selected Transformer on the same frozen MAVE subset, using its predefined schema; preserve the same evaluator and breakdowns.
2. Freeze model settings and final epoch/seed policy before either dataset's pending test evaluations.
3. Complete WDC Transformer final retraining and official test evaluation; complete MAVE final evaluation under its custom pilot protocol.
4. Implement attribute-aware normalization with an explicit canonical-value policy and independently reviewed evaluation cases.
5. Complete extraction-to-normalization raw-text inference, then the demonstration and final report.

## 9. Preservation and reproducibility

Source, configurations, summaries, small run records, validation predictions, and this documentation are saved in local Git. MAVE baseline implementation was committed as `f30b30a`; completed results were committed as `d2b88aa`. No remote Git backup has been configured.

Large datasets, environments, caches, and model weights remain outside Git in scratch. The working environment is `/scratch/csisnlp_09/envs/bert_env`; the pretrained cache is `/scratch/csisnlp_09/hf_cache`. The MAVE CRF checkpoint is `/scratch/csisnlp_09/models/mave_crf_20261009_182845_387136/model.pkl`. Selected checkpoints and required data must be preserved separately before scratch expiry.

The original [developer handover](DEVELOPER_HANDOVER.md) is retained as a historical record. This report and the current [future plan](../FUTURE_PLAN.md) describe subsequent verified progress.
