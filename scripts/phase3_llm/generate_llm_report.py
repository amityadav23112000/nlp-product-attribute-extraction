"""Independent shared pair evaluation of LLM outputs and saved baseline predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.llm_extraction.data_adapter import read_jsonl,gold_pairs
from src.llm_extraction.evaluator import evaluate_pairs
from src.llm_extraction.error_analysis import classify_errors
from src.llm_extraction.bio_entities import bio_to_entities

BASELINES={
 'CRF':'results/phase2_baselines/wdc/predictions/crf_val.jsonl',
 'BERT_seed42':'results/phase2_baselines/wdc/predictions/bert_val_seed42.jsonl',
 'DeBERTa_seed42':'results/phase2_baselines/wdc/predictions/deberta_val_seed42.jsonl'}
MAVE_BASELINES={
 'CRF':'results/phase2_baselines/mave/predictions/crf_val.jsonl',
 'DeBERTa_seed42':'results/phase2_baselines/mave/predictions/deberta_val_seed42.jsonl'}


def baseline_pairs(path,raw_by_id,ids,dataset='wdc'):
    predictions={pid:[] for pid in ids};seen=set()
    for row in read_jsonl(path):
        pid=str(row['product_id'])
        if pid not in predictions:continue
        original=raw_by_id[pid]
        if row['category']!=original['category']:raise ValueError('Baseline category mismatch')
        text=original['input_'+row['source']] if dataset=='wdc' else original['paragraphs'][int(row['id'].rsplit(':',1)[1])]['text']
        offsets=[];cursor=0
        for token in row['tokens']:
            start=text.find(token,cursor)
            if start<0 or text[cursor:start].strip():raise ValueError('Baseline tokens do not align to original source: '+row['id'])
            offsets.append((start,start+len(token)));cursor=start+len(token)
        if text[cursor:].strip():raise ValueError('Baseline omitted source text')
        for entity in bio_to_entities(row['tokens'],row['predicted']):
            left=offsets[entity['start']][0];right=offsets[entity['end']-1][1]
            predictions[pid].append(dict(attribute=entity['attribute'],value=text[left:right]))
        seen.add(pid)
    if seen!=set(ids):raise ValueError('Baseline is missing selected products')
    # BIO predicts mentions; this evaluation scores distinct product-level pairs.
    return {pid:[dict(attribute=a,value=v) for a,v in sorted({(p['attribute'],p['value']) for p in pairs})]
            for pid,pairs in predictions.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('datasets/wdc_pave'))
    p.add_argument('--dataset',choices=('wdc','mave'),default='wdc')
    a=p.parse_args()
    manifest=json.loads((a.data_dir/'manifest.json').read_text())
    ids=manifest['selected_validation_ids'];by_id={str(r['id']):r for r in read_jsonl(a.data_dir/'val.jsonl')}
    if manifest['dataset']!=a.dataset:raise ValueError('Dataset mismatch')
    gold={pid:gold_pairs(by_id[pid],a.dataset) for pid in ids};categories={pid:by_id[pid]['category'] for pid in ids}
    registry=json.loads((a.data_dir/'schema_registry.json').read_text())
    report=dict(protocol='Distinct product attribute/value pairs; exact case/spacing, Unicode NFC only; same frozen validation products',
                product_ids=ids,models={},reliability={})
    lines=['# Qwen extraction validation study','',report['protocol'],'',
           'This table recomputes shared pair metrics; it does not reuse or equate historical BIO F1. Baseline token boundaries were gold-assisted. No baseline was retrained.','',
           '| Method | Precision | Recall | Micro F1 | Macro F1 |','| --- | ---: | ---: | ---: | ---: |']
    def register(name,predictions):
        metrics=evaluate_pairs(gold,predictions,categories,registry['training_pair_counts'])
        report['models'][name]=metrics
        lines.append('| %s | %.4f | %.4f | %.4f | %.4f |'%(name,metrics['precision'],metrics['recall'],metrics['micro_f1'],metrics['macro_f1']))
        return metrics
    baselines=BASELINES if a.dataset=='wdc' else MAVE_BASELINES
    for name,path in baselines.items():register(name,baseline_pairs(path,by_id,ids,a.dataset))
    for mode in ('zero_shot','schema_guided','one_shot','three_shot'):
        folder=a.run_dir/mode
        saved=json.loads((folder/'metrics.json').read_text())
        for filename,suffix,stage in [('predictions.jsonl','raw','pre_verification'),('verified_predictions.jsonl','verified','post_verification')]:
            rows=read_jsonl(folder/filename);predictions={r['product_id']:r['attributes'] for r in rows}
            if len(rows)!=len(predictions) or set(predictions)!=set(ids):raise ValueError('Experiment prediction identity mismatch')
            metrics=register(mode+'_'+suffix,predictions)
            for key in ('precision','recall','micro_f1','macro_f1'):
                if abs(metrics[key]-saved[stage][key])>1e-12:raise ValueError('Saved versus recomputed score mismatch')
        report['reliability'][mode]=saved['reliability']
    lines+=['','## Reliability','', '| Mode | JSON parse | Unsupported | Mean latency (s) | Failures |',
            '| --- | ---: | ---: | ---: | ---: |']
    for mode,r in report['reliability'].items():
        lines.append('| %s | %.3f | %.3f | %.2f | %d |'%(mode,r['json_parse_rate'],r['unsupported_rate'],r['mean_latency_seconds'] or 0,r['generation_failures']))
    lines+=['','## Error counts after evidence verification','']
    for mode in report['reliability']:
        predictions={r['product_id']:r['attributes'] for r in read_jsonl(a.run_dir/mode/'verified_predictions.jsonl')}
        counts=Counter(e['type'] for e in classify_errors(gold,predictions))
        lines.append('- %s: %s'%(mode,json.dumps(counts)))
    lines+=['','Category/attribute/frequency results and exact FP/FN examples are in comparison.json.',
            'Exact source matching checks literal support, not whether the attribute interpretation is correct. Repeated evidence remains flagged as ambiguous.',
            'Few-shot examples come only from the restored training partition; test labels were not used. Normalization is deliberately deferred.']
    (a.run_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.run_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:19]))


if __name__=='__main__':main()
