"""CPU smoke test of the full LLM pipeline with a fake extractor (no GPU / model download).

Checks: prompt construction, JSON parsing, evidence verification, pair evaluation,
validation report against saved baselines, validation-only prompt selection and the
full 354-product test comparison - all in the repository layout.
Run:  python tests/test_llm_pipeline_cpu.py
"""
import json, sys, shutil, tempfile, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from src.llm_extraction import inference
from src.llm_extraction.data_adapter import read_jsonl


class FakeExtractor:
    """Echoes the first 3 words of the title as a PRODUCT_TYPE plus one invented value."""
    torch = None
    def generate(self, messages):
        product = json.loads(messages[-1]["content"])
        title = product["sources"][0]["text"]
        attrs = [dict(attribute="PRODUCT_TYPE", value=" ".join(title.split()[:3])),
                 dict(attribute="BRAND", value="NotInText123")]
        return dict(response=json.dumps(dict(attributes=attrs)), latency_seconds=0.0)


def main():
    tmp = Path(tempfile.mkdtemp())
    data = ROOT / "datasets/wdc_pave"
    fake = FakeExtractor()
    for mode in ("zero_shot", "schema_guided", "one_shot", "three_shot"):
        cfg = json.loads((ROOT / f"configs/llm_{mode}.json").read_text())
        m, _ = inference.run_experiment(cfg, data, tmp / "val" / mode, extractor=fake)
        assert m["reliability"]["json_parse_rate"] == 1.0
        assert m["reliability"]["unsupported_predictions"] > 0  # invented value rejected
        assert m["post_verification"]["precision"] >= m["pre_verification"]["precision"]
    import subprocess
    subprocess.run([sys.executable, "scripts/phase3_llm/generate_llm_report.py", "--run-dir", str(tmp / "val"),
                    "--data-dir", str(data)], check=True, stdout=subprocess.DEVNULL)
    report = json.loads((tmp / "val/comparison.json").read_text())
    print("validation (49 products) pair micro-F1:",
          {k: round(v["micro_f1"], 4) for k, v in report["models"].items() if "raw" not in k})
    # test stage with the fake extractor injected
    import scripts.phase3_llm.run_llm_test as rt
    original = rt.run_experiment
    rt.run_experiment = lambda *a, **k: original(*a, **k, extractor=fake)
    sys.argv = ["x", "--dataset", "wdc", "--data-dir", str(data), "--validation-dir", str(tmp / "val"),
                "--output-dir", str(tmp / "test")]
    rt.main()
    comp = json.loads((tmp / "test/test_comparison.json").read_text())
    assert comp["products"] == 354
    shutil.rmtree(tmp)
    print("OK: LLM pipeline runs end-to-end on CPU in the repository layout")


if __name__ == "__main__":
    main()
