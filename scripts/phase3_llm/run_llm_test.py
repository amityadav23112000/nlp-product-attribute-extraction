"""Select on validation only, freeze that choice, then evaluate the whole test split."""
import argparse,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.llm_extraction.inference import run_experiment,atomic_json
from src.llm_extraction.data_adapter import read_jsonl,gold_pairs
from src.llm_extraction.evaluator import evaluate_pairs
from scripts.phase3_llm.generate_llm_report import baseline_pairs


def select_prompt(validation_dir,policy,manifest):
    choices=[];common=None
    for mode in policy['mode_order']:
        root=validation_dir/mode
        metadata=json.loads((root/'run_metadata.json').read_text())
        if metadata['status']!='complete' or metadata['split']!='validation' or metadata['fingerprint']['product_ids']!=manifest['selected_validation_ids']:
            raise ValueError('Validation is incomplete or has different IDs')
        if metadata['fingerprint']['schema_sha256']!=hashlib.sha256((validation_dir/'frozen_schema.json').read_bytes()).hexdigest():
            raise ValueError('Validation schema snapshot mismatch')
        config=json.loads((root/'config.json').read_text())
        control={k:v for k,v in config.items() if k!='mode'}
        if common is None:common=control
        if control!=common or config['mode']!=mode:raise ValueError('Confounded validation configurations')
        metrics=json.loads((root/'metrics.json').read_text())
        choices.append(dict(mode=mode,validation_micro_f1=metrics['post_verification']['micro_f1'],config=config))
    selected=max(choices,key=lambda x:x['validation_micro_f1'])
    return selected,choices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('wdc','mave'),required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--validation-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    policy=json.loads(Path('configs/llm_test_policy.json').read_text())
    manifest=json.loads((a.data_dir/'manifest.json').read_text())
    if manifest['dataset']!=a.dataset or len(manifest['test_ids'])!=policy['test_scope'][a.dataset]:raise ValueError('Test scope mismatch')
    # Snapshot the exact schema used by validation, before any test generation.
    current_schema=a.data_dir/'schema_registry.json';frozen_schema=a.validation_dir/'frozen_schema.json'
    expected=hashlib.sha256(current_schema.read_bytes()).hexdigest()
    for mode in policy['mode_order']:
        metadata=json.loads((a.validation_dir/mode/'run_metadata.json').read_text())
        if metadata['fingerprint']['schema_sha256']!=expected:raise ValueError('Schema changed since validation')
        if metadata['fingerprint']['dataset_sha256']!={n:manifest['dataset_sha256'][n] for n in ('train','val')}:
            raise ValueError('Training/validation data changed since prompt selection')
    if frozen_schema.exists() and frozen_schema.read_bytes()!=current_schema.read_bytes():raise ValueError('Frozen schema changed')
    frozen_schema.write_bytes(current_schema.read_bytes())
    selected,choices=select_prompt(a.validation_dir,policy,manifest)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    selection=dict(policy=policy,dataset=a.dataset,selected=selected,validation_choices=choices,
                   validation_directory=str(a.validation_dir),test_ids=manifest['test_ids'])
    path=a.output_dir/'selection.json'
    if path.exists() and json.loads(path.read_text())!=selection:raise ValueError('Selection changed on resume')
    atomic_json(path,selection)
    metrics,_=run_experiment(selected['config'],a.data_dir,a.output_dir/'selected_prompt',a.dataset,split='test')
    # Evaluate existing final baseline predictions in the identical product/pair metric.
    rows={str(r['id']):r for r in read_jsonl(a.data_dir/'test.jsonl')};ids=manifest['test_ids']
    gold={pid:gold_pairs(rows[pid],a.dataset) for pid in ids};cats={pid:rows[pid]['category'] for pid in ids}
    registry=json.loads(current_schema.read_text())
    if a.dataset=='wdc':
        baselines={'CRF':'results/phase2_baselines/wdc/predictions/crf_test.jsonl',
                   'BERT_seed42':'results/phase2_baselines/wdc/predictions/bert_test_seed42.jsonl',
                   'DeBERTa_seed42':'results/phase2_baselines/wdc/predictions/deberta_test_seed42.jsonl'}
    else:
        baselines={'CRF':'results/phase2_baselines/mave/predictions/crf_test.jsonl',
                   'DeBERTa_seed42':'results/phase2_baselines/mave/predictions/deberta_test_seed42.jsonl'}
    report=dict(dataset=a.dataset,selected_mode=selected['mode'],products=len(ids),models={})
    for name,source in baselines.items():
        report['models'][name]=evaluate_pairs(gold,baseline_pairs(source,rows,ids,a.dataset),cats,registry['training_pair_counts'])
    report['models']['Qwen_raw']=metrics['pre_verification'];report['models']['Qwen_verified']=metrics['post_verification']
    atomic_json(a.output_dir/'test_comparison.json',report)
    lines=['# Full held-out LLM test evaluation','',f"Dataset: {a.dataset}; products: {len(ids)}; validation-selected mode: {selected['mode']}",'',
           '| Method | Precision | Recall | Micro F1 | Macro F1 |','| --- | ---: | ---: | ---: | ---: |']
    for name,m in report['models'].items():lines.append('| %s | %.4f | %.4f | %.4f | %.4f |'%(name,m['precision'],m['recall'],m['micro_f1'],m['macro_f1']))
    lines+=['','Exact product attribute/value comparison; these are not historical BIO metrics.','No test-based prompt selection; normalization deferred.']
    (a.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
