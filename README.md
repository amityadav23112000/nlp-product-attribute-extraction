# Product Attribute-Value Extraction: CRF vs Transformers vs LLMs, plus a Cross-Paradigm Committee

**CS F429 Natural Language Processing - course project (BITS Pilani, Hyderabad).**

From a noisy product title and description, extract the values of predefined attributes, for example:

```
"435952-B21 HP Xeon E5335 2.0GHz DL360 G5"
 -> PART_NUMBER=435952-B21  MANUFACTURER=HP  PROCESSOR_TYPE=Xeon E5335  CLOCK_SPEED=2.0GHz  GENERATION=G5
```

The task is framed as BIO sequence labelling and solved with models from three eras of NLP:

| Era | Model | WDC-PAVE test micro-F1 | MAVE pilot test micro-F1 |
|---|---|---:|---:|
| Classical | CRF (hand-crafted features) | 79.34 | 80.81 |
| Pre-trained Transformer | BERT-base (3 seeds) | 83.15 +- 0.18 | - |
| Pre-trained Transformer | DeBERTa-v3-base (3 seeds) | 85.77 +- 0.40 | 83.97 +- 0.44 |
| Instruction-tuned LLM | Qwen3-4B-Instruct, prompted, evidence-verified | see `results/phase3_llm/` | optional |
| **Novel (ours)** | **CPEC cross-paradigm committee** | **86.99 +- 0.09** | 83.66 (only 2 voters) |

Strict entity-level micro-F1 (seqeval). All selection is done on validation; test sets were scored once.

## Folder structure

```
datasets/      the data: WDC-PAVE raw products + generated BIO files; MAVE pilot manifests (text must be rebuilt)
src/           reusable Python package
  attrext/       BIO conversion, CRF, Transformer tagger, metrics, CPEC committee (novel method)
  llm_extraction/  Qwen prompting, JSON parsing, evidence verification, pair evaluation
  mave.py        MAVE reconstruction helpers
scripts/       runnable entry points, grouped by phase
  data/            build_wdc_bio.py
  phase2_baselines/  train_crf.py, train_transformer.py
  phase3_llm/        Qwen study: Kaggle runner, local (Mac/CUDA) runner, report/test scripts
  phase3_committee/  run_committee.py, run_committee_with_llm.py
  make_report_tables.py, make_report_figures.py
configs/       frozen hyper-parameters, prompt configs, integrity manifest
notebooks/     01-03 Qwen LLM study (Kaggle GPU), 04 live demo: training + GUI
demo/          app.py (Gradio GUI), cli.py (command line), predictor.py (shared)
results/       phase2_baselines/, phase3_llm/, phase3_committee/ - predictions, metrics, summaries
report/        main.tex (LaTeX report), main.pdf, tables/ and figures/ (auto-generated)
tests/         CPU tests for the LLM pipeline and the Transformer tagger / demo
docs/          PROJECT_STATUS.md (phases), FILE_GUIDE.md (every file), CHALLENGES.md, phase2_history/
```

A file-by-file description is in [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md); what was done in each phase is in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Quick start (CPU, about 5 minutes)

```bash
pip install -r requirements.txt
python scripts/data/build_wdc_bio.py                       # raw WDC-PAVE -> BIO (identical to Phase 2 files)
python scripts/phase2_baselines/train_crf.py --mode dev    # CRF: 78.94 val micro-F1
python scripts/phase2_baselines/train_crf.py --mode final  # CRF: 79.34 test micro-F1
python scripts/phase3_committee/run_committee.py           # CPEC: 86.99 test (WDC) from saved predictions
python tests/test_llm_pipeline_cpu.py                      # LLM pipeline smoke test with a fake model
```

## Training the Transformers (GPU: CUDA or Apple MPS)

```bash
python scripts/phase2_baselines/train_transformer.py --model deberta --mode dev   --epochs 15   # live val F1 per epoch
python scripts/phase2_baselines/train_transformer.py --model deberta --mode final --epochs 15 --seed 42
```

## LLM study (Phase 3)

* Kaggle: import `notebooks/01_phase3_qwen_llm_wdc.ipynb`, set Accelerator = GPU T4 x2 and Internet = On, set `REPO_URL`, *Save & Run All*. Download `wdc_qwen_results.zip` and unzip it into the repository root.
* Local Apple-Silicon Mac (16 GB+ RAM) or CUDA PC: `python scripts/phase3_llm/run_local.py --dataset wdc`.
* Then: `python scripts/phase3_committee/run_committee_with_llm.py && python scripts/make_report_tables.py`.

## Demo (Phase 4)

```bash
python demo/cli.py --models models --title "Xerox TONER CARTRIDGE 106R01409" --description "25K PAGE YIELD COLOR: Black"
python demo/app.py --models models --share        # Gradio GUI
```
`models/` is created by notebook 04 or by the training scripts (`crf.pkl`, `bert/`, `deberta/`). For the viva run `notebooks/04_phase4_live_demo_train_and_gui.ipynb` on Kaggle: it trains all models live, scores them on the test set and opens the GUI.

## Report

`report/main.pdf` (LaTeX source `report/main.tex`). Rebuild: `python scripts/make_report_tables.py && python scripts/make_report_figures.py && cd report && latexmk -pdf main.tex`.

## Data and licences

WDC-PAVE: Web Data Commons product attribute-value extraction benchmark (Brinkmann et al., 2024). MAVE labels: Google Research (CC BY 4.0); product text comes from the Amazon Review Data 2018 metadata and is **not** redistributed here - `notebooks/02_phase3_prepare_mave_pilot.ipynb` rebuilds the exact frozen pilot from hashed manifests.
