# File guide - what is in every folder and file

## `datasets/`
| Path | Content |
|---|---|
| `wdc_pave/train.jsonl`, `val.jsonl`, `test.jsonl` | Raw WDC-PAVE products (906 / 160 / 354). Fields: `id`, `cluster_id`, `category`, `input_title`, `input_description`, `target_scores` (attribute -> value -> `pid` = 0 title / 1 description; `n/a` = absent). |
| `wdc_pave/manifest.json` | SHA-256 of the three splits, product IDs, the 49-product LLM validation sample and 10-product smoke sample. |
| `wdc_pave/schema_registry.json` | Attribute inventory per category, attribute definitions used in schema-guided prompts, training pair counts (frequency bands). |
| `wdc_pave/bio/{split}.jsonl` | Generated BIO sequences (one per title/description): `tokens`, `labels`, ids, category, source. Made by `scripts/data/build_wdc_bio.py`. |
| `wdc_pave/bio/*_unmatched.jsonl`, `*_overlaps.jsonl`, `stats.json` | Gold values not found in text, overlapping annotations dropped, corpus statistics. |
| `mave_pilot/selection_manifest.json` | The frozen 5,000 MAVE product IDs and their train/val/test split and duplicate groups. |
| `mave_pilot/label_schema.json`, `mave_label_profile.json` | 28 attributes / 57 BIO labels and label statistics. |
| `mave_pilot/preparation_report.json` | Full provenance of the pilot reconstruction (counts, hashes, exclusions). |
| `mave_pilot/mave_release_manifest.json`, `mave_metadata_manifest.json` | Pinned MAVE label release and Amazon metadata archive (URL, size, SHA-256). |
| `mave_pilot/official_cleaner/` | Google's official MAVE metadata cleaner (vendored, licence included). |

## `src/` (library code)
| File | Content |
|---|---|
| `attrext/bio.py` | WDC products -> BIO (value matching, one-occurrence selection, boundary-forcing tokenizer); `tokenize_raw` for real inference. |
| `attrext/crf.py` | CRF features (word, affixes, shape, flags, neighbours, source) and train/predict. |
| `attrext/transformer_tagger.py` | BERT / DeBERTa token classifier: first-sub-word alignment, training loop (AdamW, linear decay, clipping, AMP), prediction, save/load. Works on CUDA, Apple MPS or CPU. |
| `attrext/metrics.py` | seqeval entity metrics, BIO <-> spans, frequency bands, BIO -> (attribute, value) pairs. |
| `attrext/committee.py` | **Novel method CPEC**: span/pair voting with per-band thresholds and validation-only threshold selection. |
| `attrext/io.py` | JSON/JSONL helpers. |
| `llm_extraction/prompt_builder.py` | System prompt and zero-shot / schema-guided / 1-shot / 3-shot message construction. |
| `llm_extraction/model_loader.py` | Loads pinned Qwen3-4B-Instruct on CUDA (1 or 2 GPUs) or Apple MPS; greedy generation with memory/latency logging. |
| `llm_extraction/output_parser.py` | Strict JSON/schema parsing with issue logging. |
| `llm_extraction/evidence_verifier.py` | Keeps only values that literally occur in the product text (exact / ambiguous / unsupported). |
| `llm_extraction/inference.py` | Resumable experiment runner with per-product atomic checkpoints and metrics. |
| `llm_extraction/evaluator.py`, `error_analysis.py` | Product-level pair metrics and error classes. |
| `llm_extraction/data_adapter.py`, `schema_registry.py`, `bio_entities.py`, `runtime.py` | Public product view (no gold leakage), demonstrations, schema building, BIO->entities, compute backends (slurm/kaggle/local). |
| `mave.py` | MAVE metadata streaming (nested gzip), evidence validation, BIO conversion. |

## `scripts/` (entry points)
| File | What it does |
|---|---|
| `data/build_wdc_bio.py` | Builds `datasets/wdc_pave/bio/`. |
| `phase2_baselines/train_crf.py` | CRF dev (train->val) or final (train+val->test). Reproduces 78.94 / 79.34. |
| `phase2_baselines/train_transformer.py` | Fine-tunes BERT or DeBERTa (dev or final), saves checkpoint, predictions, history. |
| `phase3_llm/kaggle_runner.py` | Used by the Kaggle notebooks: integrity check, GPU check, model download, stages, results zip, MAVE preparation. |
| `phase3_llm/run_local.py` | Same study on a local Mac (MPS) or CUDA PC. |
| `phase3_llm/run_llm_ablation.py`, `run_llm_extraction.py`, `run_llm_test.py` | Smoke + 4-prompt validation study; validation-only prompt selection and full test run. |
| `phase3_llm/generate_llm_report.py`, `evaluate_llm.py` | Validation comparison with saved CRF/BERT/DeBERTa predictions in the same pair metric. |
| `phase3_llm/recover_mave_llm_data.py` | Rebuilds the MAVE pilot text from the pinned sources. |
| `phase3_llm/make_integrity_manifest.py` | Regenerates `configs/llm_integrity_manifest.json`. |
| `phase3_committee/run_committee.py` | CPEC on WDC and MAVE from saved predictions (entity level). |
| `phase3_committee/run_committee_with_llm.py` | CPEC with Qwen as 4th voter (pair level). |
| `make_report_tables.py`, `make_report_figures.py` | Regenerate `report/tables/*.tex` and `report/figures/*`. |

## `configs/`
`bert_selected.json`, `deberta_selected.json` (selected development settings), `wdc_final_policy.json`, `mave_final_policy.json` (frozen final-training rules), `mave_subset.json`, `mave_deberta_baseline.json`, `llm_*.json` (four prompting conditions and the test policy), `llm_integrity_manifest.json` (hashes checked before every LLM run).

## `notebooks/`
`01_phase3_qwen_llm_wdc.ipynb` (Kaggle GPU, WDC LLM study), `02_phase3_prepare_mave_pilot.ipynb` (CPU, rebuild MAVE text), `03_phase3_qwen_llm_mave.ipynb` (MAVE LLM study), `04_phase4_live_demo_train_and_gui.ipynb` (viva demo: live training + test table + CLI + GUI).

## `results/`
| Path | Content |
|---|---|
| `phase2_baselines/wdc/predictions/` | Saved token predictions (`tokens`, `gold`, `predicted`) of CRF, BERT and DeBERTa on validation (dev models) and test (final models), seeds 42/43/44. |
| `phase2_baselines/wdc/reports/` | Phase 2 summaries: CRF grid search, BERT/DeBERTa tuning and diagnoses, validation comparisons, final test summary. |
| `phase2_baselines/wdc/rerun_crf_{dev,final}/` | Fresh CRF re-runs with the cleaned code (metrics, predictions, model). |
| `phase2_baselines/mave/` | Same for the MAVE pilot (CRF, DeBERTa 3 seeds). |
| `phase3_committee/{wdc,mave}/` | CPEC results: `summary.md`, `results.json` (all seeds, error profile, band scores, selection table), committee test predictions. |
| `phase3_llm/` | Qwen outputs after the run: raw responses, parsed/verified predictions, metrics, selection, test comparison. |
| `phase3_committee/wdc_with_llm/` | CPEC + Qwen (after the LLM run). |

## Other
`demo/` GUI/CLI, `report/` LaTeX report, `tests/` CPU tests, `docs/phase2_history/` original Phase 2 working notes (kept for provenance; paths there refer to the old HPC layout).
