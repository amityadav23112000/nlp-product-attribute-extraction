"""Separate public product text from gold annotations and training demonstrations."""
import hashlib
import json
from pathlib import Path
import re


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def attribute_name(raw):
    return re.sub(r'[^A-Z0-9]+','_',raw.upper()).strip('_')


def public_product(row, dataset='wdc'):
    if dataset=='wdc':
        sources=[dict(source=key,text=row.get(field,'') or '') for key,field in
                 [('title','input_title'),('description','input_description')]]
    else:
        sources=[dict(source=str(p.get('source','paragraph'))+'_'+str(i),text=p['text'])
                 for i,p in enumerate(row['paragraphs'])]
    return dict(product_id=str(row['id']),category=row['category'],sources=sources)


def gold_pairs(row,dataset='wdc'):
    if dataset=='wdc':
        return [dict(attribute=attribute_name(a),value=v) for a,values in row['target_scores'].items()
                for v in values if v.casefold()!='n/a']
    # Reconstructed MAVE product annotations retain released evidence values.
    return [dict(attribute=attribute_name(a['key']),value=v)
            for a in row.get('attributes',[]) for v in sorted({e['value'] for e in a.get('evidences',[])})]


def stratified_sample(rows,per_category,seed=42):
    selected=[]
    for category in sorted({r['category'] for r in rows}):
        candidates=[r for r in rows if r['category']==category]
        candidates.sort(key=lambda r:hashlib.sha256(('%d:%s'%(seed,r['id'])).encode()).hexdigest())
        selected.extend(candidates[:per_category])
    return selected


def demonstrations(training,category,count,seed=42,dataset='wdc'):
    candidates=[r for r in training if r['category']==category]
    candidates.sort(key=lambda r:hashlib.sha256(('%d:%s'%(seed,r['id'])).encode()).hexdigest())
    if len(candidates)<count:
        raise ValueError('Not enough training demonstrations in '+category)
    return [dict(product=public_product(r,dataset),answer=dict(attributes=gold_pairs(r,dataset)))
            for r in candidates[:count]]
