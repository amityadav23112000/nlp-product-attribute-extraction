"""Modern baseline: fine-tuned Transformer token classifier (BERT / DeBERTa-v3).

Architecture: pre-trained encoder (bert-base-uncased: 12 layers, 768 hidden, 12 heads,
WordPiece; deberta-v3-base: 12 layers, 768 hidden, 12 heads, disentangled
content/position attention, SentencePiece 128k vocab) + dropout + linear layer over
2K+1 BIO labels. All encoder weights are updated (full fine-tuning).
Supervision: only the FIRST sub-word of each word carries the word's BIO label; other
sub-words get -100 (ignored by the loss). Word prediction = first sub-word argmax.

Selected Phase 2 settings (validation-only selection): AdamW lr 5e-5, weight decay 0.01
(not on bias/LayerNorm), batch 16, linear decay without warm-up, gradient clipping 1.0,
max length 512 (no truncation allowed), bf16/fp16 autocast on GPU.
Final models: train+validation, fixed epoch budget (BERT 14, DeBERTa 15), seeds 42/43/44.
"""
import json
import time
from pathlib import Path

PINNED = {"microsoft/deberta-v3-base": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
          "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594"}


def label_schema(records):
    attrs = sorted({l[2:] for r in records for l in r["labels"] if l != "O"})
    labels = ["O"] + [p + a for a in attrs for p in ("B-", "I-")]
    return {l: i for i, l in enumerate(labels)}


def align_labels(word_ids, labels, label2id):
    out, previous = [], None
    for wid in word_ids:
        out.append(-100 if wid is None or wid == previous else label2id[labels[wid]])
        previous = wid
    return out


class TransformerTagger:
    def __init__(self, model_name, label2id=None, revision=None, device=None, checkpoint=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        source = checkpoint or model_name
        revision = None if checkpoint else (revision or PINNED.get(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, revision=revision)
        if checkpoint:
            self.model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        else:
            id2label = {i: l for l, i in label2id.items()}
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name, revision=revision, num_labels=len(label2id), label2id=label2id, id2label=id2label,
                ignore_mismatched_sizes=True)
        self.model.to(self.device)
        self.label2id = self.model.config.label2id
        self.id2label = {int(i): l for i, l in self.model.config.id2label.items()}

    def _encode(self, records, with_labels=True):
        feats = []
        for r in records:
            enc = self.tokenizer(r["tokens"], is_split_into_words=True, truncation=False)
            if len(enc["input_ids"]) > 512:
                raise ValueError("sequence longer than 512 sub-words: " + str(r.get("id")))
            item = dict(enc)
            if with_labels:
                item["labels"] = align_labels(enc.word_ids(), r["labels"], self.label2id)
            feats.append(item)
        return feats

    def train(self, records, epochs=15, lr=5e-5, batch_size=16, weight_decay=0.01, seed=42,
              eval_records=None, log=print):
        torch = self.torch
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForTokenClassification, get_linear_schedule_with_warmup, set_seed
        set_seed(seed)
        loader = DataLoader(self._encode(records), batch_size=batch_size, shuffle=True,
                            collate_fn=DataCollatorForTokenClassification(self.tokenizer),
                            generator=torch.Generator().manual_seed(seed))
        decay = [p for n, p in self.model.named_parameters() if not (n.endswith("bias") or "LayerNorm.weight" in n)]
        no_decay = [p for n, p in self.model.named_parameters() if n.endswith("bias") or "LayerNorm.weight" in n]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)
        sched = get_linear_schedule_with_warmup(opt, 0, len(loader) * epochs)
        use_amp = self.device == "cuda"
        dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
        history = []
        for epoch in range(1, epochs + 1):
            self.model.train(); losses = []; t0 = time.time()
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast(self.device, dtype=dtype, enabled=use_amp):
                    loss = self.model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                losses.append(loss.item())
            row = dict(epoch=epoch, train_loss=sum(losses) / len(losses), seconds=round(time.time() - t0, 1))
            if eval_records is not None:
                from .metrics import entity_scores
                row.update(entity_scores([r["labels"] for r in eval_records], self.predict(eval_records)))
            history.append(row)
            log(json.dumps(row))
        return history

    def predict(self, records, batch_size=32):
        torch = self.torch
        from transformers import DataCollatorWithPadding
        self.model.eval(); out = []
        collate = DataCollatorWithPadding(self.tokenizer)
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            encs = [self.tokenizer(r["tokens"], is_split_into_words=True, truncation=False) for r in chunk]
            batch = collate([dict(e) for e in encs])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                ids = self.model(**batch).logits.argmax(-1).cpu().tolist()
            for enc, row, r in zip(encs, ids, chunk):
                labels, prev = [], None
                for pos, wid in enumerate(enc.word_ids()):
                    if wid is not None and wid != prev:
                        labels.append(self.id2label[row[pos]])
                    prev = wid
                assert len(labels) == len(r["tokens"])
                out.append(labels)
        return out

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path); self.tokenizer.save_pretrained(path)
