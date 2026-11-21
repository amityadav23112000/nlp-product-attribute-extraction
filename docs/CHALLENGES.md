# Challenges and solutions

1. **Noisy product text** (run-together words like `BlackALTERNATE`, part numbers, repeated values). Boundary-forcing tokenisation; deterministic one-occurrence labelling; unmatched values (63) and overlapping annotations (197) are logged, not silently dropped.
2. **BERT first scored below the CRF** (63% vs 79% validation F1). Audited sub-word alignment, padding masks, metrics; a bounded learning-rate/epoch study fixed it (80.6%, lr 5e-5, up to 15 epochs).
3. **DeBERTa's SentencePiece tokenizer** differs from WordPiece. Separate alignment audit and native/fast tokenizer equivalence check; model revisions pinned.
4. **MAVE ships only labels.** Product text had to be re-joined with 12 GB of Amazon metadata that was gzip-compressed twice; the first reconstruction failed. Fixed with a nested-gzip reader with CRC checks, SHA-256 manifests and a frozen 5,000-product pilot with group-aware splits.
5. **Test-set discipline.** Every decision (CRF regularisation, LR, epochs, prompt, CPEC thresholds) uses validation only; final epoch budgets fixed in advance; test scored once.
6. **LLM hallucination / invalid JSON.** Strict parser, schema checks and exact evidence verification; LLM votes count only when grounded in the text.
7. **Compute.** The HPC allocation ended; Qwen-4B needs ~8 GB of weights. Portable runners for Kaggle T4 x2 (model sharding, fp16) and Apple-Silicon MPS with resumable per-product checkpoints.
8. **Messy research repository** (hard-coded scratch paths, SLURM logs, failed runs). Restructured into datasets/src/scripts/results/report with path-independent code; Phase 2 numbers re-verified bit-for-bit from the cleaned code (BIO files identical; CRF 78.94/79.34; DeBERTa 85.77; BERT 83.15).
9. **Negative results.** A frequency-routed CRF/DeBERTa hybrid and the 2-voter MAVE committee did not help; analysis showed that a vote needs at least three *different* voters, which led to CPEC.
