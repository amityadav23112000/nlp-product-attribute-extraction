"""Separate exact pair errors from source-support diagnostics."""
from collections import Counter
from .evaluator import pair_key


def classify_errors(gold,predictions):
    output=[]
    for pid,expected in gold.items():
        g=Counter(set(pair_key(p) for p in expected));p=Counter(pair_key(x) for x in predictions[pid])
        matched=g&p;missing=sorted((g-matched).elements());extra=sorted((p-matched).elements())
        for attr,value in missing:
            alternative=next((x for x in extra if x[1]==value),None)
            kind='incorrect_attribute'
            if alternative is None:
                alternative=next((x for x in extra if x[0]==attr),None)
                kind='incorrect_value'
            if alternative:
                extra.remove(alternative)
                output.append(dict(product_id=pid,type=kind,gold=dict(attribute=attr,value=value),
                                   prediction=dict(attribute=alternative[0],value=alternative[1])))
            else:output.append(dict(product_id=pid,type='complete_miss',gold=dict(attribute=attr,value=value)))
        for attr,value in extra:
            output.append(dict(product_id=pid,type='spurious_extraction',prediction=dict(attribute=attr,value=value)))
    return output
