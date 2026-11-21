"""One-to-one product attribute/value metrics, separate from historical BIO F1."""
from collections import Counter,defaultdict
import unicodedata


def pair_key(item):
    # Exact value after Unicode NFC only: no lowercasing, unit or whitespace repair.
    return item['attribute'],unicodedata.normalize('NFC',item['value'])


def score(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,tp=tp,fp=fp,fn=fn,support=tp+fn)


def evaluate_pairs(gold,predictions,categories,frequency=None):
    if set(gold)!=set(predictions):
        raise ValueError('Prediction product IDs must match gold, including failures')
    counts=defaultdict(Counter);cat_counts=defaultdict(Counter);errors=[]
    for pid in gold:
        expected=Counter(set(pair_key(p) for p in gold[pid]))
        predicted=Counter(pair_key(p) for p in predictions[pid])
        for attr,value in sorted(expected.keys()|predicted.keys()):
            tp=min(expected[attr,value],predicted[attr,value]);fp=predicted[attr,value]-tp;fn=expected[attr,value]-tp
            for target in (counts[attr],cat_counts[categories[pid]]):target.update(tp=tp,fp=fp,fn=fn)
            if fp or fn:errors.append(dict(product_id=pid,attribute=attr,value=value,false_positive=fp,false_negative=fn))
    per_attribute={a:score(c['tp'],c['fp'],c['fn']) for a,c in sorted(counts.items())}
    result=score(*(sum(c[k] for c in counts.values()) for k in ('tp','fp','fn')))
    result.update(macro_f1=sum(v['micro_f1'] for v in per_attribute.values())/len(per_attribute) if per_attribute else 0.,
                  per_attribute=per_attribute,per_category={a:score(c['tp'],c['fp'],c['fn']) for a,c in sorted(cat_counts.items())},products=len(gold),errors=errors)
    if frequency is not None:
        bucket=lambda n:'head' if n>100 else 'medium' if n>=20 else 'tail' if n>=5 else 'extreme_tail'
        result['per_frequency_group']={}
        for group in ('head','medium','tail','extreme_tail'):
            chosen=[c for a,c in counts.items() if bucket(frequency.get(a,0))==group]
            result['per_frequency_group'][group]=score(*(sum(c[k] for c in chosen) for k in ('tp','fp','fn')))
    return result
