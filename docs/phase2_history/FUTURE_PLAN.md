# Remaining project plan

Updated: 9 October 2026. Based on the supplied developer handover and inspected code/results. This is a plan; no new experiments are launched by it.

Progress update: phase 1's shared validation audit/error reports and annotation-independent tokenization foundation are complete. Phase 2's three-LR study and selected-config confirmation at seeds 42/43/44 are complete: mean validation micro F1 0.805991, sample standard deviation 0.006035, versus CRF 0.7894. See `results/bert_tuning_summary.md`, `results/bert_validation_diagnosis.md`, and `configs/bert_selected.json`. No official test evaluation has occurred. Next: phase 3 (DeBERTa), then freeze final retraining rules before phase 4. Full raw-text model inference remains part of phase 5.

Subsequent update: phase 3 is complete. DeBERTa-v3-base passed alignment and native/fast-tokenizer compatibility audits. Five validation-only experiments selected LR 5e-5; confirmation seeds 42/43/44 averaged 0.844409 micro F1 with sample standard deviation 0.003995. All five runs passed common metric/dataset-hash verification. Selected settings and model revision are in `configs/deberta_selected.json`; reports are in `results/deberta_tuning_summary.md` and `results/deberta_validation_diagnosis.md`. Next: phase 4, first freezing the final retraining epoch/seed policy, then final retraining and official test evaluation.

User-directed priority change: proceed with MAVE before WDC final testing. The user supplied an authorized full Amazon Review Data (2018) metadata URL. Official MAVE label files are verified; metadata download/index job 386785 is active. The tested reconstruction job 386812 is queued after successful completion. The initial bounded protocol and pending outputs are documented in `docs/MAVE.md`. Full MAVE corpus audits and model experiments remain pending. WDC final test evaluation and normalization are still outstanding.

MAVE completion update: job 386785 completed download/indexing. The first reconstruction failed because the archive was gzip-compressed twice. After fixing and testing the reader, job 387113 successfully prepared `pilot_v2` with 5,000 products (train 3,548; validation 736; test 716), 28 predefined attributes and 57 BIO labels. Corpus audits, independent split-file hashes and product/group separation checks passed. Next establish the MAVE CRF validation baseline, then train Transformers on the identical files using the predefined label schema. No MAVE model predictions have been evaluated yet.

## Verified starting point

Update 10 October: WDC final Transformer evaluation completed as job 387830, exit 0. BERT official test mean micro F1 0.831501 (sample SD 0.001771), DeBERTa 0.857705 (SD 0.004000). Independent prediction/metric/fixed-history/aggregate checks passed. WDC and custom MAVE pilot final experiments are complete. The next planned phase is attribute-aware normalization and raw-text inference; a full official MAVE experiment requires a separately specified reconstruction/benchmark protocol.

Latest completion: final MAVE retraining/test evaluation passed. CRF test micro F1 0.808132; DeBERTa three-seed mean 0.839723, sample standard deviation 0.004438. The earlier MAVE training/testing tasks below are satisfied. Next resume WDC Transformer final retraining and official testing under frozen policies; normalization and the raw-input demonstration remain pending.

MAVE confirmation update: seeds 43 and 44 are complete; the three-seed mean validation micro F1 is 0.826198 with sample standard deviation 0.004711. All three exceed CRF. Next freeze final epoch/seed/reporting policies before held-out testing; do not select a final seed from its validation score. Test evaluation and normalization remain pending.

Latest completion: MAVE DeBERTa validation baseline is complete (micro F1 0.828521, best epoch 10, stopped at epoch 13). The test split remains untouched. The earlier immediate-baseline instructions are now satisfied; next freeze final retraining/epoch/seed policies before pending final evaluations, or specify additional validation-only experiments if needed. See the saved same-split CRF/DeBERTa comparison.

Current completion update: the MAVE CRF baseline completed as job 387136 with validation micro F1 0.805266 and macro F1 0.688001. Independent metric and error reconciliation checks passed; the test split remains untouched. Earlier progress paragraphs above record historical stages. See [docs/WORK_COMPLETED.md](docs/WORK_COMPLETED.md) for the consolidated current record. MAVE Transformer validation is the immediate next experiment.

- WDC-PAVE preprocessing and CRF development/final evaluation are complete.
- Products: train 906, validation 160, official test 354. Keep these splits and processed gold files fixed for existing benchmark comparisons.
- CRF validation micro F1: 0.7894; official test micro F1: 0.7934.
- BERT training, GPU/BF16 verification, subword alignment, prediction export, and tokenizer audits are implemented and have passed.
- The rerun in `artifacts/bert_20261009_151511_386708` completed five epochs, with validation micro F1 0.630588 at epoch 5.
- A separate completed run exists at `/scratch/csisnlp_09/models/bert_wdc_run_005`, with validation micro F1 0.631610. Register its configuration and provenance before choosing between runs.
- BERT official test evaluation, DeBERTa, normalization, MAVE, and a working inference demo remain outstanding in the inspected project.
- The working BERT environment and pretrained cache are `/scratch/csisnlp_09/envs/bert_env` and `/scratch/csisnlp_09/hf_cache`. Historical handover environment/queue details are stale.

## 1. Consolidate experiment records and audit BERT

Deliverables: experiment index, shared evaluation command, validation comparison and error report.

1. Record every completed run's configuration, seed, dataset hashes, checkpoint, metrics, versions, elapsed time, and location. Distinguish failed attempts from completed runs.
2. Confirm that CRF and BERT validation predictions contain the same IDs, token sequences, and gold labels. Recompute their strict entity metrics with one shared evaluator and explicit BIO handling.
3. Parameterize the existing error-analysis scripts, which currently hard-code CRF prediction paths. Preserve one-to-one entity matching; verify false-positive/false-negative totals against the evaluator.
4. Compare misses, boundaries, attribute confusions, and spurious spans by source, category, and attribute. Prioritize PRODUCT_TYPE, MATERIAL, BRAND/MANUFACTURER, identifiers, and rare attributes.
5. Inspect illegal BIO predictions, absent training I-labels, padding/loss masks, scheduler behavior, classifier gradients, and training/validation learning curves. Use a small training-only memorization check if a defect is suspected. Repair confirmed defects before tuning; do not assume the score gap proves an implementation bug.
6. Document that current benchmark token boundaries depend on gold spans. Preserve this historical benchmark, but implement a separate tokenizer for raw-text inference that requires no annotations. Do not describe gold-assisted tokenization as deployable raw-text performance.

Exit criterion: an auditable comparison and evidence explaining the main BERT losses, with alignment and metric correctness confirmed.

## 2. Tune and freeze BERT using development data

Deliverables: bounded experiment matrix, selected configuration, learning curves and seed summary.

- Retain the current run as the initial baseline.
- Use a small staged search: learning rates 1e-5, 2e-5, and 5e-5 at batch size 16; allow up to 15 epochs with validation-based early stopping. The current best is at epoch 5, so the five-epoch cutoff merits investigation but does not establish undertraining.
- After choosing the learning rate/epoch range, test warmup and batch size only if diagnostics justify them. Avoid changing several factors at once.
- Keep first-subword supervision and strict entity micro F1 as the baseline protocol. Any BIO-constrained decoding or alternative loss must be a named ablation with validation-only selection.
- Repeat the selected configuration with three predetermined seeds and report mean/spread. GPU reruns are not guaranteed to match exactly even with the same seed.
- Freeze architecture, preprocessing, decoding, seed/reporting policy, hyperparameters, and final epoch rule before looking at official test predictions.

Exit criterion: a reproducible BERT configuration with a justified selection procedure. Outperforming CRF is a hypothesis, not a requirement for a valid experiment.

## 3. Establish DeBERTa on the same WDC-PAVE split

Deliverables: validation metrics, predictions, error analysis and selected configuration.

- Reuse the audited trainer/evaluator for the handover's DeBERTa-v3-base candidate.
- Validate this model's own tokenizer and alignment rather than assuming BERT WordPiece behavior; check dependencies and sequence lengths in the allocated GPU environment.
- Use a similarly bounded validation search and the same comparison tables and seed policy.
- Preserve identical processed gold, product splits, and primary metric across model families.

Exit criterion: a frozen DeBERTa configuration and fair validation comparison.

## 4. Final WDC-PAVE evaluation and long-tail analysis

Deliverables: final model checkpoints, official test predictions, comparative results tables, uncertainty estimates.

- Implement final retraining separately from development training. Combine 906 training and 160 validation products; train for an epoch budget determined from development runs. Never select checkpoints using the official test set.
- Freeze long-tail thresholds using development training positive entity-span counts: head >100, medium 20–100, tail 5–19, extreme tail <5. Count spans, not positive tokens. Define zero-support handling explicitly.
- Evaluate the frozen BERT and DeBERTa final models once per predetermined seed policy on 354 test products, preserving the completed CRF result.
- Report precision/recall, micro/macro F1, source/category/attribute breakdowns, and frequency-group performance with support counts. Use paired product-level resampling if confidence intervals are included, keeping title and description together.
- Keep conclusions grounded in results, including a possible CRF advantage. No tuning in response to test scores.

Exit criterion: completed RQ1 and initial WDC-PAVE evidence for RQ3.

## 5. Normalization and the complete inference pipeline

Deliverables: attribute-aware normalizer, annotated evaluation cases, extraction-to-normalization CLI.

- Start with high-value numeric attributes: capacity, dimensions, clock speed, rotational speed, pack quantity, page yield, and weights where the schema supports them.
- Define canonical units, decimal versus binary storage conventions, rounding, accepted aliases, and ambiguous-input behavior before measuring performance. For example, 8192 MB must not automatically become 8 GB without an explicit binary convention.
- Preserve raw values, normalized values, original text offsets, and normalization status; leave unsupported or ambiguous values unchanged.
- Build reviewed reference cases from development data with held-out cases for evaluation. Do not invent canonical gold labels from existing BIO labels.
- Evaluate normalization separately on gold extracted values, then end to end on predicted values. Keep canonical-value accuracy separate from strict raw-span F1.
- Build a CLI that accepts raw title/description, runs annotation-independent tokenization and extraction, reconstructs exact source substrings, and returns structured JSON plus a readable display.
- Choose the deployed extractor from development evidence; allow CRF as a fallback if it remains stronger. Add a GUI only after the CLI works.

Exit criterion: a usable extraction-plus-normalization system and measurable RQ2 results.

## 6. MAVE reconstruction and controlled experiments

Deliverables: documented source/access feasibility, deterministic reconstruction manifest, fixed subset and splits, comparative results.

- First verify the actual public release, required metadata, access terms, and reconstructable fields against primary dataset documentation. The handover's reconstruction description is an assumption to verify before downloading large archives.
- Measure successful joins, missing text, span recovery, duplicate identities and leakage. Retain stable product IDs and report excluded records.
- Choose a manageable category/attribute subset and compute/storage budget before full reconstruction. Split by product and relevant duplicate/group identity; freeze this protocol before experiments.
- Reuse validated data adapters and evaluation; run CRF and the selected Transformer on the same MAVE subset. Document label/schema mappings before any cross-dataset transfer claim.
- Report within-MAVE head/tail comparisons. Do not treat CRF on WDC-PAVE versus BERT on MAVE as a model comparison.

Exit criterion: credible second-dataset evidence for RQ3. If access or deadline prevents completion, report MAVE as deferred rather than claiming completed experiments.

## 7. Submission and preservation

Deliverables: README, environment records, reproducible commands, final report, slides and demo examples.

- Keep source, configs, experiment summaries and reports in persistent storage. Keep growing caches/checkpoints in scratch and preserve selected final checkpoints before scratch expiry; do not duplicate every run into the home quota.
- Use allocated SLURM accelerator resources for training/heavy work and the documented course MPS allocation. Check live jobs before launching another run; do not reuse historical job IDs blindly.
- Report processed-BIO limitations, gold-assisted benchmark tokenization, overlaps, non-contiguous values, repeated mentions, class imbalance and small supports.
- Prepare results tables while phases complete, not only at the end. Separate extraction quality from normalization quality and benchmark quality from raw-input demo behavior.

## Priority and dependencies

Immediate: same-subset Transformer validation experiments on the prepared MAVE `pilot_v2` files; the CRF baseline is complete. Resume WDC final retraining/test evaluation afterward using the already selected settings and a policy frozen before test predictions.

Main experiment chain: 1 → 2 → 3 → 4.

System chain: annotation-independent inference work in 1 → 5. Normalization specification and evaluation-case preparation can proceed while Transformer jobs run.

Second-dataset chain: stable WDC-PAVE pipeline → 6. A small MAVE feasibility check can happen earlier, but large reconstruction should not delay the working primary system.

Documentation and preservation in 7 continue throughout. Calendar dates require the submission deadline and available GPU budget; this plan uses completion gates instead of invented dates.
