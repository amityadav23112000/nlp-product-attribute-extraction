"""Recompute pair metrics for a generated/verified validation prediction file."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.llm_extraction.data_adapter import read_jsonl,gold_pairs
from src.llm_extraction.evaluator import evaluate_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('datasets/wdc_pave'))
    p.add_argument('--dataset',choices=('wdc','mave'),default='wdc')
    p.add_argument('--split',choices=('validation',),default='validation')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    rows=read_jsonl(a.predictions);predicted={r['product_id']:r['attributes'] for r in rows}
    if len(predicted)!=len(rows):raise ValueError('Duplicate prediction IDs')
    manifest=json.loads((a.data_dir/'manifest.json').read_text())
    if manifest['dataset']!=a.dataset:raise ValueError('Dataset differs from prepared manifest')
    expected=set(manifest['selected_validation_ids'])
    if set(predicted)!=expected and set(predicted)!=set(manifest['smoke_ids']):raise ValueError('Incomplete or unexpected prediction sample')
    data={str(r['id']):r for r in read_jsonl(a.data_dir/'val.jsonl')}
    gold={pid:gold_pairs(data[pid],a.dataset) for pid in predicted};cats={pid:data[pid]['category'] for pid in predicted}
    result=evaluate_pairs(gold,predicted,cats)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))},indent=2))


if __name__=='__main__':main()
