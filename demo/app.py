"""Gradio GUI: paste a product title/description and compare CRF, BERT, DeBERTa and the
CPEC committee side by side, with extracted values highlighted in the text.

  python demo/app.py --models models/ [--share]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.predictor import Extractor

EXAMPLES = [
    ["435952-B21 HP Xeon E5335 2.0GHz DL360 G5", "Intel Xeon E5335 DL360 G5(2.0GHz/4-core/8MB-2x4MB/80W) Full Processor Option Kit"],
    ["Xerox TONER CARTRIDGE FOR WORKCENTRE 4250 - 106R01409", "25K PAGE YIELD COLOR: Black UPC: 095205742466"],
    ["Seagate Barracuda 2TB 7200 RPM SATA 6Gb/s 3.5-Inch Internal Hard Drive", "64MB cache, ST2000DM001"],
]


def build(extractor):
    import gradio as gr

    def run(title, description):
        rows = extractor.extract(title, description)
        systems = list(extractor.models) + (["CPEC committee"] if len(extractor.models) >= 2 else [])
        tables = [extractor.as_table(rows, s) or [["-", "-", "-"]] for s in systems]
        main = "CPEC committee" if "CPEC committee" in systems else systems[-1]
        spans = sorted((r for r in rows if r["system"] == main and r["source"] == "title"), key=lambda r: r["start"])
        hl, cur = [], 0
        for r in spans:
            if r["start"] < cur:
                continue
            hl += [(title[cur:r["start"]], None), (title[r["start"]:r["end"]], r["attribute"])]
            cur = r["end"]
        hl.append((title[cur:], None))
        return [hl] + tables

    systems = list(extractor.models) + (["CPEC committee"] if len(extractor.models) >= 2 else [])
    with gr.Blocks(title="Product attribute extraction") as ui:
        gr.Markdown("## Product attribute-value extraction - CS F429 NLP project\nClassical CRF vs fine-tuned Transformers vs the novel cross-paradigm committee (CPEC).")
        with gr.Row():
            title = gr.Textbox(label="Product title", lines=2)
            desc = gr.Textbox(label="Product description", lines=4)
        go = gr.Button("Extract attributes", variant="primary")
        highlight = gr.HighlightedText(label="Title, highlighted by the committee", combine_adjacent=False)
        outs = []
        with gr.Row():
            for s in systems:
                outs.append(gr.Dataframe(headers=["attribute", "value", "source"], label=s, wrap=True))
        go.click(run, [title, desc], [highlight] + outs)
        gr.Examples(EXAMPLES, [title, desc])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models"); ap.add_argument("--freq"); ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    ex = Extractor(a.models, json.loads(Path(a.freq).read_text()) if a.freq else None)
    build(ex).launch(share=a.share)
