"""Resumable LLM experiments with atomic per-product records and audited outputs."""
from collections import Counter
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from .data_adapter import read_jsonl,public_product,gold_pairs,demonstrations
from .prompt_builder import build_messages
from .output_parser import parse_response
from .evidence_verifier import verify_attributes
from .evaluator import evaluate_pairs
from .error_analysis import classify_errors


def _version(package):
    try:return version(package)
    except Exception:return None


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def run_experiment(config,data_dir,output_dir,dataset='wdc',smoke=False,extractor=None,split='validation'):
    data_dir=Path(data_dir);output_dir=Path(output_dir)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if manifest['dataset']!=dataset:raise ValueError('Dataset differs from prepared manifest')
    if split not in ('validation','test') or (split=='test' and smoke):raise ValueError('Invalid evaluation split/smoke combination')
    evaluation_name='val' if split=='validation' else 'test'
    required_names=('train',evaluation_name)
    for name in required_names:
        if hashlib.sha256((data_dir/(name+'.jsonl')).read_bytes()).hexdigest()!=manifest['dataset_sha256'][name]:
            raise ValueError('Prepared data hash changed')
    registry=json.loads((data_dir/'schema_registry.json').read_text())
    train=read_jsonl(data_dir/'train.jsonl');all_rows=read_jsonl(data_dir/(evaluation_name+'.jsonl'))
    ids=manifest['test_ids'] if split=='test' else manifest['smoke_ids' if smoke else 'selected_validation_ids']
    by_id={str(r['id']):r for r in all_rows}
    if len(by_id)!=len(all_rows):raise ValueError('Duplicate validation products')
    rows=[by_id[pid] for pid in ids]
    if split=='test' and set(ids)!=set(by_id):raise ValueError('Test evaluation must include every held-out product')
    if set(ids)&{str(r['id']) for r in train}:raise ValueError('Demonstration leakage')
    config_hash=hashlib.sha256(json.dumps(dict(config=config,dataset=dataset,smoke=smoke,split=split),sort_keys=True).encode()).hexdigest()
    fingerprint=dict(configuration_sha256=config_hash,dataset_sha256={n:manifest['dataset_sha256'][n] for n in required_names},product_ids=ids,
                     schema_sha256=hashlib.sha256((data_dir/'schema_registry.json').read_bytes()).hexdigest())
    if os.environ.get('KAGGLE_RUNTIME_PROFILE'):
        fingerprint['runtime_profile']=json.loads(os.environ['KAGGLE_RUNTIME_PROFILE'])
    output_dir.mkdir(parents=True,exist_ok=True)
    meta_path=output_dir/'run_metadata.json'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        if previous['fingerprint']!=fingerprint:raise ValueError('Refusing to mix configurations/data on resume')
        metadata=previous
    else:
        metadata=dict(status='running',fingerprint=fingerprint,model=config['model'],revision=config['revision'],
                      dataset=dataset,split=split,smoke=smoke,mode=config['mode'],python=platform.python_version(),
                      versions={p:_version(p) for p in ('torch','transformers','accelerate')},
                      allocation=os.environ.get('SLURM_JOB_ID'),execution_backend=os.environ.get('LLM_EXECUTION_BACKEND','slurm'),
                      started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      test_loaded=split=='test',normalization='deferred',demonstration_policy='Training-only, deterministic category ranking; one-shot is a prefix of three-shot')
        atomic_json(output_dir/'config.json',config)
        atomic_json(meta_path,metadata)
    journal=output_dir/'products';journal.mkdir(exist_ok=True)
    count={'zero_shot':0,'schema_guided':0,'one_shot':1,'three_shot':3}[config['mode']]
    for index,row in enumerate(rows,1):
        public=public_product(row,dataset);pid=public['product_id'];path=journal/(pid+'.json')
        examples=demonstrations(train,public['category'],count,config['seed'],dataset)
        messages,allowed,demo_ids=build_messages(public,registry,config['mode'],examples)
        prompt_hash=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()
        if path.exists():
            if json.loads(path.read_text())['prompt_sha256']!=prompt_hash:raise ValueError('Prompt changed during resume')
            continue
        if extractor is None:
            from .model_loader import QwenExtractor
            extractor=QwenExtractor(config)
        result=dict(product_id=pid,category=public['category'],demonstration_ids=demo_ids,prompt_sha256=prompt_hash,
                    generation_failed=False,retry_count=0)
        try:
            generated=extractor.generate(messages)
            result.update(generated)
        except (RuntimeError,ValueError) as error:
            if 'out of memory' in str(error).lower():extractor.empty_cache()
            result.update(generation_failed=True,error=str(error),response='',latency_seconds=None)
        result['parsed']=parse_response(result['response'],allowed)
        result['raw_predictions'],result['verified_predictions']=verify_attributes(result['parsed']['attributes'],public)
        atomic_json(path,result)
        print('%s %d/%d product=%s parsed=%s accepted=%d'%(config['mode'],index,len(rows),pid,result['parsed']['structure_valid'],len(result['verified_predictions'])),flush=True)
    # Gold is accessed only after generation; prompt builder receives public fields.
    records=[json.loads((journal/(pid+'.json')).read_text()) for pid in ids]
    gold={str(r['id']):gold_pairs(r,dataset) for r in rows};categories={str(r['id']):r['category'] for r in rows}
    pre={r['product_id']:r['raw_predictions'] for r in records};post={r['product_id']:r['verified_predictions'] for r in records}
    metrics={stage:evaluate_pairs(gold,pred,categories,registry['training_pair_counts']) for stage,pred in [('pre_verification',pre),('post_verification',post)]}
    raw_count=sum(len(r['raw_predictions']) for r in records)
    unsupported=sum(p['evidence_status']=='unsupported' for r in records for p in r['raw_predictions'])
    latencies=[r['latency_seconds'] for r in records if r['latency_seconds'] is not None]
    issue_counts=Counter(i['type'] for r in records for i in r['parsed']['issues'])
    generated_items=sum(r['parsed']['generated_items'] for r in records)
    metrics['reliability']=dict(products=len(records),generation_failures=sum(r['generation_failed'] for r in records),
          json_parse_rate=sum(r['parsed']['json_valid'] for r in records)/len(records),
          schema_structure_rate=sum(r['parsed']['structure_valid'] for r in records)/len(records),
          strict_json_rate=sum(r['parsed']['json_valid'] and not r['parsed']['fence_removed'] for r in records)/len(records),
          unsupported_predictions=unsupported,raw_predictions=raw_count,unsupported_rate=unsupported/raw_count if raw_count else 0.,
          ambiguous_evidence=sum(p['evidence_status']=='ambiguous' for r in records for p in r['raw_predictions']),
          mean_latency_seconds=sum(latencies)/len(latencies) if latencies else None,total_generation_seconds=sum(latencies),
          peak_gpu_bytes=max((r.get('peak_gpu_bytes',0) for r in records),default=0),issue_counts=dict(issue_counts),
          out_of_schema_rate=issue_counts['out_of_schema']/generated_items if generated_items else 0.,
          reached_token_limit=sum(r.get('reached_token_limit',False) for r in records),retry_count=0)
    errors=classify_errors(gold,post)
    for filename,items in [('raw_responses.jsonl',records),('predictions.jsonl',[dict(product_id=pid,attributes=pre[pid]) for pid in ids]),
                           ('verified_predictions.jsonl',[dict(product_id=pid,attributes=post[pid]) for pid in ids]),('errors.jsonl',errors)]:
        (output_dir/filename).write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in items))
    atomic_json(output_dir/'metrics.json',metrics)
    atomic_json(output_dir/'per_category_metrics.json',metrics['post_verification']['per_category'])
    atomic_json(output_dir/'per_attribute_metrics.json',metrics['post_verification']['per_attribute'])
    metadata.update(status='failed' if metrics['reliability']['generation_failures']==len(records) else 'complete',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    gpu=extractor.device_names() if hasattr(extractor,'device_names') else metadata.get('gpu'))
    atomic_json(meta_path,metadata)
    if metrics['reliability']['generation_failures']==len(records):raise RuntimeError('Every model generation failed; inspect raw records')
    return metrics,extractor
