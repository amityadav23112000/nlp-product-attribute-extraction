"""Raw product text -> attribute/value table, using the trained CRF / BERT / DeBERTa and the
CPEC committee. Shared by the Gradio GUI (demo/app.py) and the CLI (demo/cli.py).
Tokenisation here is annotation-independent (no gold boundaries), i.e. real deployment."""
import pickle, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attrext.bio import tokenize_raw
from src.attrext.crf import predict_crf
from src.attrext.committee import vote_spans
from src.attrext.metrics import entities


class Extractor:
    def __init__(self, model_dir, freq=None, tau=None):
        model_dir = Path(model_dir)
        self.models = {}
        if (model_dir / "crf.pkl").exists():
            self.models["CRF"] = ("crf", pickle.loads((model_dir / "crf.pkl").read_bytes()))
        for name in ("bert", "deberta"):
            if (model_dir / name).exists():
                from src.attrext.transformer_tagger import TransformerTagger
                self.models[name.upper() if name == "bert" else "DeBERTa"] = ("tf", TransformerTagger(None, checkpoint=str(model_dir / name)))
        self.freq = freq or {}
        self.tau = tau or {"head": 2, "medium": 2, "tail": 2, "extreme_tail": 3}

    def _label(self, kind, model, records):
        return predict_crf(model, records) if kind == "crf" else model.predict(records)

    def extract(self, title, description=""):
        rows = []
        for source, text in (("title", title or ""), ("description", description or "")):
            toks = tokenize_raw(text)
            if not toks:
                continue
            record = dict(id=source, source=source, tokens=[t["text"] for t in toks], labels=["O"] * len(toks))
            preds = {name: self._label(kind, m, [record])[0] for name, (kind, m) in self.models.items()}
            if len(preds) >= 2:
                preds["CPEC committee"] = vote_spans(list(preds.values()), self.freq, self.tau)
            for system, labels in preds.items():
                for attr, s, e in sorted(entities(labels), key=lambda x: x[1]):
                    rows.append(dict(system=system, source=source, attribute=attr,
                                     value=text[toks[s]["start"]:toks[e]["end"]], start=toks[s]["start"], end=toks[e]["end"]))
        return rows

    @staticmethod
    def as_table(rows, system):
        return [[r["attribute"], r["value"], r["source"]] for r in rows if r["system"] == system]
