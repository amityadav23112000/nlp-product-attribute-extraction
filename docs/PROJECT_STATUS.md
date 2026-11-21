# Project status: what was done in each phase and what remains

Course: CS F429 Natural Language Processing. Task: **product attribute-value extraction** (BIO sequence labelling).
Datasets: **WDC-PAVE** (main) and a reconstructed **MAVE** 5,000-product pilot (second dataset).

| Phase | Course part | Deadline | Status |
|---|---|---|---|
| 1 | Part 1: problem, dataset, plan (1-2 page report) | 28 Aug 2026 | **Done** |
| 2 | Part 2: baselines + report sections a-d + demo | 12 Oct 2026 | **Done** |
| 3 | Part 3: LLM study + novel method + improvement | 22 Nov 2026 | Novel method **done**; Qwen LLM run see below |
| 4 | Part 3: full report, challenges, final demo (GUI/CLI) | 22 Nov 2026 | **Done** (report auto-updates when LLM results are added) |

## Phase 1 (done)
Problem definition, motivation, dataset choice (WDC-PAVE + MAVE), plan. History: `docs/phase2_history/`.

## Phase 2 (done) - classical vs modern baselines
- WDC-PAVE preprocessing to BIO (906/160/354 products, 51 attributes, 103 labels).
- CRF (classical): validation 78.94, **test 79.34** micro-F1. Reproduced exactly by `scripts/phase2_baselines/train_crf.py`.
- BERT-base: learning-rate/epoch study, 3 seeds; **test 83.15 +- 0.18**.
- DeBERTa-v3-base: same protocol; **test 85.77 +- 0.40**.
- MAVE pilot reconstruction (5,000 products, 3,548/736/716), CRF **80.81**, DeBERTa **83.97 +- 0.44**.
- Shared evaluator, error analysis (miss / spurious / boundary / wrong attribute), frequency bands.
Evidence: `results/phase2_baselines/`.

## Phase 3 (this submission) - LLM + novel contribution
1. **Instruction-tuned LLM** Qwen3-4B-Instruct-2507: zero-shot, schema-guided, 1-shot, 3-shot prompts on 49 validation
   products -> validation-only prompt selection -> all 354 test products, with exact **evidence verification**.
   Code: `src/llm_extraction/`, `scripts/phase3_llm/`; run with `notebooks/01_phase3_qwen_llm_wdc.ipynb` (Kaggle GPU)
   or `scripts/phase3_llm/run_local.py` (Apple-Silicon Mac / CUDA PC). Results go to `results/phase3_llm/`.
2. **Novel method - CPEC (Cross-Paradigm Evidence-grounded Committee)**: CRF + BERT + DeBERTa (+ Qwen) vote on each
   candidate with frequency-band thresholds chosen on validation. **WDC test 86.99 +- 0.09 vs DeBERTa 85.77 +- 0.40**
   (+1.22, every seed improves, 36% fewer spurious and 43% fewer boundary errors).
   Code: `src/attrext/committee.py`, `scripts/phase3_committee/`. Results: `results/phase3_committee/`.
3. Negative results (reported honestly): frequency-routed CRF/DeBERTa hybrid (85.60 < 85.77); 2-voter MAVE committee (83.66 < 83.97).

## Phase 4 (this submission) - final system, report, demo
- `notebooks/04_phase4_live_demo_train_and_gui.ipynb`: live training of CRF, BERT, DeBERTa (epoch-by-epoch F1),
  test scoring incl. CPEC, then CLI + **Gradio GUI**.
- `demo/app.py` (GUI) and `demo/cli.py` (CLI) on raw text with annotation-free tokenisation.
- Report: `report/main.tex` -> `report/main.pdf` (LaTeX, research-paper format, challenges section).

## Remaining steps before submission
1. Run the Qwen WDC study (notebook 01 on Kaggle or `run_local.py` on the Mac); copy `wdc_qwen_results.zip` contents
   into `results/phase3_llm/`.
2. `python scripts/phase3_committee/run_committee_with_llm.py` (CPEC with Qwen as 4th voter).
3. `python scripts/make_report_tables.py && cd report && latexmk -pdf main.tex` (tables refresh automatically).
4. Fill group names/IDs on the report title page; push to GitHub.
5. Optional: MAVE LLM run (notebooks 02-03 or `run_local.py --dataset mave --prepare`).
