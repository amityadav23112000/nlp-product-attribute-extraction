"""Run the Phase 3 Qwen study on a LOCAL machine (Apple-Silicon Mac via MPS, or a CUDA PC).

  python scripts/phase3_llm/run_local.py --dataset wdc                # validation (4 prompts) + test
  python scripts/phase3_llm/run_local.py --dataset wdc --stage smoke  # 10-product check first
  python scripts/phase3_llm/run_local.py --dataset mave --prepare     # rebuild MAVE pilot, then run

Same frozen configs, data hashes, prompts and evaluation as the Kaggle notebooks. Results are
written to results/phase3_llm/<dataset>_{validation,test}/ and are resumable: rerun the same
command after an interruption and finished products are skipped.
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
import kaggle_runner as runner


def preflight():
    import torch
    from src.llm_extraction.runtime import accelerator
    device = accelerator(torch)
    x = torch.ones(1, device=device) + 1
    info = dict(device=device, torch=torch.__version__)
    if device == "mps":
        import subprocess
        info["ram_gb"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
        if info["ram_gb"] < 16:
            raise RuntimeError("Qwen3-4B in float16 needs ~8 GB of weights; at least 16 GB unified memory is required.")
    from importlib.metadata import version
    os.environ["KAGGLE_RUNTIME_PROFILE"] = json.dumps(dict(gpus=[device], dtype="float16" if device == "mps" else "auto",
        versions={p: version(p) for p in ("torch", "transformers", "accelerate")}), sort_keys=True)
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("wdc", "mave"), default="wdc")
    ap.add_argument("--stage", choices=("smoke", "validation", "test", "all"), default="all")
    ap.add_argument("--prepare", action="store_true", help="MAVE only: download + rebuild the frozen pilot first")
    a = ap.parse_args()
    os.environ["LLM_EXECUTION_BACKEND"] = "local"
    os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(Path.home() / ".cache/huggingface/hub")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    runner.configure()
    runner.verify_bundle()
    if a.dataset == "mave" and a.prepare:
        runner.prepare_mave()
    runner.check_data(a.dataset)
    preflight()
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    cfg = json.loads((HERE.parents[1] / "configs/llm_zero_shot.json").read_text())
    snapshot_download(cfg["model"], revision=cfg["revision"], cache_dir=os.environ["HF_HUB_CACHE"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"])
    t0 = time.time()
    stages = ["validation", "test"] if a.stage == "all" else [a.stage]
    for stage in stages:
        runner.run(a.dataset, stage)
    print(f"done in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
