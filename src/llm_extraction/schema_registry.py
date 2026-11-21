"""Category ontologies use training schema keys, never evaluation labels."""
from collections import Counter
from .data_adapter import attribute_name,gold_pairs

DEFINITIONS={
 'HEIGHT':'Vertical dimension; require explicit context identifying height. Do not guess axis order.',
 'WIDTH':'Width dimension; distinguish it from height, depth and length using explicit context.',
 'DEPTH':'Front-to-back dimension; do not infer it solely from an unlabeled dimension sequence.',
 'LENGTH':'Length dimension; require support for this axis rather than guessing width or height.',
 'BRAND':'Brand explicitly identifying the offered product, not a compatibility target.',
 'MANUFACTURER':'Explicit maker of the product; do not automatically equate with brand.',
 'COLOR':'A color value for the COLOR attribute; keep separate from COLOR_S.',
 'COLOR_S':'The Color(s) attribute in this ontology; do not rename it COLOR.',
 'CAPACITY':'Explicit storage or holding capacity; distinguish capacity from package count.',
 'PACK_QUANTITY':'Number of items in the offered pack, box or carton, not storage capacity.',
 'PRODUCT_TYPE':'Literal product type phrase from the text, not a paraphrased category.',
 'INTERFACE':'Explicit hardware connection/interface type.',
 'ROTATIONAL_SPEED':'Explicit rotational speed, preserving its source spelling and units.',
 'CLOCK_SPEED':'Explicit processor clock frequency, preserving source spelling and units.',
 'BUS_SPEED':'Explicit system bus speed, not processor clock speed.',
 'PART_NUMBER':'Exact part identifier; preserve punctuation and capitalization.',
 'MANUFACTURER_STOCK_NUMBER':'Exact manufacturer stock identifier; preserve punctuation.',
 'RETAIL_UPC':'Exact UPC identifier; preserve leading zeros.',
 'MATERIAL':'Explicit material of the offered product, not an inferred constituent.',
 'SIZE_WEIGHT':'Explicit size/weight value under the Size/Weight field; preserve original units.',
 'PAPER_WEIGHT':'Explicit paper weight; do not reinterpret basis weight as product mass.',
 'PAGE_YIELD':'Explicit printing yield, not model numbers or pack size.'}


def build_registry(training,dataset='wdc'):
    categories={};counts=Counter()
    for row in training:
        names=[attribute_name(a) for a in row['target_scores']] if dataset=='wdc' else [attribute_name(a['key']) for a in row.get('attributes',[])]
        categories.setdefault(row['category'],set()).update(names)
        counts.update(p['attribute'] for p in gold_pairs(row,dataset))
    attributes=sorted({a for names in categories.values() for a in names})
    return dict(categories={c:sorted(names) for c,names in categories.items()},attributes=attributes,
                definitions={a:DEFINITIONS.get(a,'Extract the literal '+a.lower().replace('_',' ')+' value; omit it if not explicitly supported.') for a in attributes},
                training_pair_counts=dict(counts))
