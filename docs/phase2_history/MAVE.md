# MAVE preparation

Updated: 9 October 2026. This stage prepares a controlled second-dataset pilot; no MAVE model has been trained yet.

Reconstruction is now complete in `pilot_v2`: 5,000 products, 61,118 paragraph sequences and 1,046,166 tokens. Successful job 387113 exited with code 0. The saved BIO audits report no unknown labels, length errors or invalid transitions. Independent file hashing and product/group overlap checks also passed. The output schema contains 28 predefined attributes and 57 BIO labels.

| Split | Products | Sequences | Tokens |
| --- | ---: | ---: | ---: |
| Train | 3548 | 43302 | 743574 |
| Validation | 736 | 8789 | 150642 |
| Test | 716 | 9027 | 151950 |

There are exactly 1,000 selected products each for Shoes, Shirts & Tops, Watches, Books and Dresses. The complete scan read 15,023,059 metadata records, matched all 15,000 unique candidates, and retained 14,725 valid candidates before final selection. Exclusions were 267 overlength products, seven tokenizer coverage failures and one evidence-offset mismatch. Flat BIO conversion rejected 620 overlapping evidence spans in the selected subset. Some predefined attributes have no positive training examples; report these supports rather than changing splits in response to held-out data.

The first preparation job 386812 failed: this supplied archive contains two gzip layers, and the original reader decoded only one. Its report is retained as `results/mave_preparation_failed_v1.json`, and its incomplete directory is preserved. The corrected reader detects gzip magic bytes recursively, rejects non-JSON payloads before scanning, and validates both gzip CRCs at EOF. All 19 current MAVE tests passed on the compute node, including nested-gzip and corrupted-inner-CRC cases.

## Sources

The [official MAVE release](https://github.com/google-research-datasets/MAVE) provides positive/negative attribute evidence labels and a deterministic cleaner, rather than the product paragraphs. The [Amazon Review Data (2018) documentation](https://nijianmo.github.io/amazon/index.html) describes the required product metadata. The user supplied an authorized URL for the full `All_Amazon_Meta.json.gz` archive from `mcauleylab.ucsd.edu`.

The official repository is pinned to `f84cefbd71578d4a84a70664ebdfc8f09b62910e`. Small upstream files and the accompanying license are preserved under `vendor/mave/`. The labels are downloaded through Git LFS media URLs and checked against the exact pointer sizes and SHA-256 hashes. They are not committed to the project repository.

Verified label files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Positives | 744855753 | `2338610c0d6cbbd7247786d942a3e13c0938b903f3cc75ffa77cd98d0d4b2187` |
| Negatives | 154939405 | `9b20f4448b6301a5fb4885d6d1436983258b22c4e6de6dab86a748e366fd1ed1` |

The metadata archive is 12,237,944,088 bytes (11.4 GiB compressed). It is downloaded to scratch with resume support and a local SHA-256 manifest. Reconstruction reads gzip directly; reaching a valid EOF also checks its gzip CRC. A complete uncompressed copy is not created.

## Workflow history

- Job **386785** completed metadata download and label indexing.
- Job **386812** ran after successful download/index completion but failed on the nested-gzip input.
- Corrected job **387113** completed reconstruction successfully into `pilot_v2`.
- The earlier label-only job 386780 failed at a network connection reset. The corrected fetcher uses retrying curl transfers.

Logs are in `results/mave_metadata_download_386785.log`, `results/mave_download_386785.log` and `results/mave_prepare_387113.log`. Runtime logs remain on disk and are ignored by Git. No further download or reconstruction is needed for this prepared pilot.

All heavy work runs inside the institute's course SLURM allocation. Both jobs use four CPUs, 32 GiB memory and the required MPS resource request. The preparation job does not launch model training.

## Frozen initial pilot protocol

The initial configuration is in `configs/mave_subset.json`:

1. Index positives and negatives by ASIN using SQLite; preserve attribute keys and evidence arrays. Contradictory positive/negative labels are excluded and counted during reconstruction.
2. Choose five categories by positive-label product count before metadata reconstruction, unless categories are explicitly configured. This targets a pilot, not broad representative category sampling.
3. Choose a bounded candidate pool of up to 3,000 products per category by the smallest seeded SHA-256 ranks. Stream the complete metadata archive, parsing full records only for candidate ASINs.
4. Run the exact pinned official source conversion and cleaner. Preserve paragraph order: title, description, feature, price, brand, including multiple paragraphs per source.
5. Require each evidence's paragraph, character offsets and value to match cleaned text exactly. Do not repair offsets with fuzzy matching. Unrepresentable whitespace boundaries are rejected explicitly.
6. Use the WDC benchmark's base regex with evidence-boundary splitting, retain all explicit non-overlapping evidence occurrences, and record annotations rejected by flat BIO overlap resolution.
7. Check coverage and the 512-subword limit with both cached BERT and DeBERTa tokenizers. Exclude whole overlength products and count these exclusions; no gold entity is truncated. This introduces a documented short-text selection bias. If fewer than 200 valid products remain in a chosen category, stop with an exclusion report rather than silently changing the protocol.
8. Select up to 1,000 valid products per category by their original ranks: at most 5,000 products overall.
9. Group exact normalized title/description/feature duplicates and known parent-ASIN families transitively. Assign complete groups to train/validation/test using a stable hash with 70%/15%/15% thresholds. These are expected proportions, not exact stratified counts.
10. Audit product/group/duplicate-text disjointness, BIO validity and label membership. Save immutable split files, selected IDs, source hashes, configuration/code hashes, overlap logs and final statistics.

The raw attribute-name mapping is preserved, with deterministic disambiguation of normalization collisions. A predefined label inventory is established from public candidate annotations before splitting. The shared checker/trainer accept `--label-schema` for that fixed inventory; WDC retains its existing training-derived schema when the argument is absent.

## Outputs

Raw assets: `/scratch/csisnlp_09/data/mave/raw/`.

Label index: `/scratch/csisnlp_09/data/mave/interim/labels.sqlite`.

Prepared pilot: `/scratch/csisnlp_09/data/mave/processed/pilot_v2/`:

- `train.jsonl`, `val.jsonl`, `test.jsonl`: same core token/BIO/product/category/source fields consumed by the current training/evaluation tools.
- `label_schema.json`: predefined output schema and original attribute-name mapping.
- `products.jsonl`: selected reconstructed products with original evidence offsets.
- `selection_manifest.json`: immutable product IDs, groups and sampling frontier.
- `rejected_overlaps.jsonl`: discarded flat-BIO overlaps.
- `preparation_report.json`: final data counts, audits, exclusions, provenance and hashes.

Small reports are also saved under `results/`: `mave_release_manifest.json`, `mave_metadata_manifest.json`, `mave_label_profile.json`, and `mave_preparation_report.json`. A dataset is ready only when the preparation report says `status: prepared`; an existing output directory alone is not evidence of completion.

The running first index job loaded the script before the embedded-provenance enhancement. Its source checksums remain recorded in the completed profile; the preparation script verifies those checksums against the release manifest and verifies database product counts before accepting that legacy index.

## Validation and subsequent experiments

Final held-out evaluation completed successfully: job `387301`, exit 0, elapsed 1h15m57s. On the 716-product custom pilot test, CRF micro F1 is 80.81%; DeBERTa seeds 42/43/44 score 83.61%/83.85%/84.47%. DeBERTa mean is 83.97%, sample standard deviation 0.44 percentage points, improving over CRF by 3.16 percentage points. Independent saved-prediction identity/gold checks, test hash, exact metrics, fixed 10-epoch histories and aggregate recomputation passed. Test was loaded after all models finished training, with no test-based checkpoint selection. Reports: `artifacts/mave_final_20261009_210933_387301/summary.md` and `test_results.json`; checkpoints remain in the corresponding scratch model directory. WDC Transformer official test evaluation remains pending.

Final policy frozen in `configs/mave_final_policy.json`: retrain on train plus validation (4,284 products), DeBERTa seeds 42/43/44, exactly 10 epochs (median development best epoch), final checkpoint only, no early stopping, and a linear schedule over the fixed final budget. CRF uses its unchanged 150-iteration configuration on the same combined training data. `scripts/mave_final.py` trains every final model before loading test, verifies frozen split hashes and training/test separation, then evaluates each model once on the 716-product custom pilot test. Frequency groups retain original development-training counts. Reports include all seeds, aggregate mean/spread and source/category/attribute/error breakdowns; no seed or setting is selected from test scores. Launcher: `scripts/mave_final.sbatch`. This is a custom MAVE final evaluation, separate from pending WDC official test work.

Three-seed confirmation is complete: job `387172` exited 0 after 44m32s. Seeds 42/43/44 achieved validation micro F1 82.85%/82.93%/82.08%, with best epochs 10/13/6. Mean 82.62%; sample standard deviation 0.47 percentage points. All seeds exceed the CRF 80.53%, and the mean improvement is 2.09 percentage points. Configuration equality except seed, dataset/schema/model revision agreement, best-epoch consistency and shared validation/error checks passed. Summary statistics were independently recomputed. The test split remains untouched. See `results/mave_confirmation_387172/summary.md` and its `comparison/` directory. Earlier submission paragraphs below are historical.

Confirmation job `387172` was submitted to run seeds 43 and 44 sequentially with the unchanged seed-42 baseline configuration. Launcher: `scripts/mave_confirm.sbatch`; orchestration and summary: `scripts/confirm_mave_seeds.py`. The seed-aware baseline launcher supplies only the changed seed. After both complete, provenance/configuration and best-epoch checks precede a shared validation comparison with seed 42 and CRF. Mean and sample standard deviation will be written to `results/mave_confirmation_387172/summary.json` and `summary.md`. These are pending results; the test split remains untouched.

Completion update for job `387147`: COMPLETED, exit 0, 34m51s elapsed. DeBERTa stopped after epoch 13 with best checkpoint epoch 10. Validation precision 0.794799, recall 0.865231, micro F1 0.828521, macro F1 0.708722; micro F1 improves over CRF by 2.325454 percentage points. Shared prediction identity/gold checks, split separation, error reconciliation and saved history/checkpoint consistency passed. Complete misses decreased from 242 to 150, while spurious predictions increased from 258 to 294. This is one seed, not a multi-seed estimate. No test evaluation occurred. Records: `artifacts/mave_deberta_20261009_184522_387147/`; comparison: `results/mave_deberta_20261009_184522_387147/comparison.md`. The submission paragraph below records the earlier launch status.

Transformer baseline job `387147` was submitted on 9 October 2026 using `scripts/mave_transformer.sbatch` and `configs/mave_deberta_baseline.json`. It uses the selected, pinned WDC DeBERTa-v3-base settings (LR 5e-5, seed 42, batch 16, maximum 15 epochs, patience 3) as a single MAVE development baseline, without a MAVE hyperparameter search. The launcher verifies frozen train/validation hashes, supplies the predefined schema, keeps checkpoints in scratch, and compares completed validation predictions with the CRF automatically. Submission is not evidence of training completion; no Transformer score is available yet. The MAVE CRF reference score is supplied explicitly rather than reusing the historical WDC reference.

Nineteen MAVE regression/integration tests passed in the allocated preparation environment, including official HTML cleaning/source order, offset failures, overlap handling, positive/negative merging, label collisions, transitive grouping, stable candidate sampling, predefined schema behavior, nested-gzip decoding and inner-CRC failures. Full corpus preparation and independent saved-file audits have now completed successfully.

The user confirmed continuing with the 5,000-product subset. CRF development training completed successfully as SLURM job `387136` on 2026-10-09 (exit 0, 7m42s allocation elapsed, approximately 2.1 GiB peak memory). Its run directory is `artifacts/mave_crf_20261009_182845_387136/`. Validation precision is 0.801341, recall 0.809231, micro F1 0.805266 and macro F1 0.688001. The fixed baseline uses the WDC lexical/shape/neighbor/source features, L-BFGS, c1=0.1, c2=0.1 and completed 150 iterations. Configuration: `configs/mave_crf_baseline.json`; reproducible launcher: `scripts/mave_crf.sbatch`.

The launcher trains on 3,548 products (43,302 paragraph sequences), evaluates only the 736-product validation split (8,789 sequences), and automatically writes exact-span metrics, predictions and the shared category/source/attribute/error analysis. Frozen dataset hashes, train/validation group separation, independent seqeval metric agreement and error-count reconstruction all passed. Reports are in `results/mave_crf_20261009_182845_387136/`. Weights are saved under `/scratch/csisnlp_09/models/`; the 716-product test split was not loaded. Errors include 242 complete misses, 258 spurious predictions, 47 attribute confusions and 21 boundary errors. Source F1 varies from 0.8969 for title to 0.7013 for feature; rare attributes require support-aware interpretation.

After the CRF result is verified, train the selected Transformer on the identical MAVE train/validation files, freezing settings before evaluating this pilot's test split. The generic Transformer study runner now accepts `--data-dir` and `--label-schema` as well as `--base-config`.

This is a custom bounded MAVE experiment, not a reproduction of the publisher's original train/test or zero-shot benchmark. Missing attribute annotations do not prove absence; the flat BIO gold uses released observed evidence. Overlapping annotations and near-duplicate variants remain limitations. Token boundaries are gold-assisted, so benchmark scores must not be described as raw-input deployment performance. No automatic equivalence between MAVE and WDC attribute schemas is assumed.
