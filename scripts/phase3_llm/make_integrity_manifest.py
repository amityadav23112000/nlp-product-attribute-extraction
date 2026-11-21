"""Write configs/llm_integrity_manifest.json: SHA-256 of every file the LLM run depends on.
The Kaggle runner refuses to start if any of them changed (protects frozen data/configs)."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FILES = (sorted((ROOT / "src/llm_extraction").glob("*.py")) + [ROOT / "src/__init__.py", ROOT / "src/mave.py", ROOT / "src/attrext/__init__.py", ROOT / "src/attrext/bio.py"] + sorted((ROOT / "scripts/phase3_llm").glob("*.py"))
         + sorted((ROOT / "configs").glob("llm_*.json"))
         + [ROOT / "datasets/wdc_pave" / n for n in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "schema_registry.json")]
         + sorted((ROOT / "results/phase2_baselines/wdc/predictions").glob("*seed42.jsonl"))
         + [ROOT / "results/phase2_baselines/wdc/predictions/crf_val.jsonl", ROOT / "results/phase2_baselines/wdc/predictions/crf_test.jsonl",
            ROOT / "requirements_kaggle_llm.txt"])
files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in FILES if p.name not in ("make_integrity_manifest.py", "llm_integrity_manifest.json") and "__pycache__" not in str(p)}
manifest = dict(format_version=2, model="Qwen/Qwen3-4B-Instruct-2507",
                model_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                base_git_commit="repository layout v2", files=files)
(ROOT / "configs/llm_integrity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("hashed", len(files), "files")
