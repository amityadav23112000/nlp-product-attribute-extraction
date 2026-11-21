"""MAVE reconstruction, evidence validation and immutable flat-BIO conversion."""
import ast
from collections import Counter
import copy
from contextlib import contextmanager, ExitStack
import gzip
import hashlib
import itertools
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import typing

from src.attrext.bio import normalize_label, tokenize_with_boundaries, assign_bio


UPSTREAM_REVISION = 'f84cefbd71578d4a84a70664ebdfc8f09b62910e'
CLEANER_SHA256 = '7c8e4969e34d08bbac5c1df7e04ba8d1b88725de84e88eaa745a569b74d9725e'


@contextmanager
def metadata_stream(path):
    """Stream plain or nested gzip JSONL and reject non-JSON payloads early."""
    with ExitStack() as stack:
        stream = stack.enter_context(Path(path).open('rb'))
        layers = 0
        while stream.peek(2)[:2] == b'\x1f\x8b':
            if layers >= 3:
                raise ValueError('Metadata has more than three gzip layers')
            stream = stack.enter_context(gzip.GzipFile(fileobj=stream, mode='rb'))
            layers += 1
        if not stream.peek(256).lstrip().startswith(b'{'):
            raise ValueError('Decoded metadata payload is not JSONL objects')
        yield stream, layers


def official_cleaner(source_path=None):
    """Load only reviewed cleaning definitions, without Beam/flags/pipeline setup.

    The upstream function bodies remain unchanged. Beam's execution base and
    counters are supplied locally; HTML parsing still uses BeautifulSoup.
    """
    import bs4
    source_path = source_path or Path(__file__).resolve().parents[1] / 'datasets/mave_pilot/official_cleaner/clean_amazon_product_metadata_main.py'
    content = Path(source_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != CLEANER_SHA256:
        raise ValueError('Official cleaner checksum differs from reviewed revision')
    names = {'ConvertSourcesFn', 'CleanParagraphsFn', 'is_css', 'is_html', 'remove_tags'}
    constants = {'_TITLE', '_DESCRIPTION', '_FEATURE', '_PRICE', '_BRAND', '_SOURCES', '_JsonObject'}
    nodes = []
    for node in ast.parse(content).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    if {node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))} != names:
        raise ValueError('Reviewed cleaning definitions are missing')
    counters = Counter()
    class Metric:
        def __init__(self, name):
            self.name = name
        def inc(self, amount=1):
            counters[self.name] += amount
    beam = SimpleNamespace(DoFn=object, metrics=SimpleNamespace(
        Metrics=SimpleNamespace(counter=lambda namespace, name: Metric(name))))
    namespace = dict(copy=copy, itertools=itertools, json=__import__('json'), re=re,
                     logging=logging, bs4=bs4, beam=beam)
    namespace.update({name: getattr(typing, name) for name in ('Any', 'Iterator', 'Mapping', 'Sequence', 'Tuple', 'Union')})
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source_path), 'exec'), namespace)
    converter, cleaner = namespace['ConvertSourcesFn'](), namespace['CleanParagraphsFn']()
    def clean(metadata, labels):
        converted = next(converter.process((metadata, labels)))
        return next(cleaner.process(converted), None)
    return clean, counters


def merge_attributes(positive, negative):
    attributes = {}
    for polarity, rows in (('positive', positive), ('negative', negative)):
        for attribute in rows:
            key = attribute['key']
            evidence = attribute.get('evidences', [])
            if not isinstance(key, str) or not key or not isinstance(evidence, list):
                raise ValueError('Invalid attribute schema')
            if polarity == 'negative' and evidence:
                raise ValueError('Negative annotation contains positive evidence')
            if polarity == 'positive' and not evidence:
                raise ValueError('Positive annotation has no evidence')
            if key in attributes:
                raise ValueError('Duplicate or contradictory attribute: ' + key)
            attributes[key] = copy.deepcopy(attribute)
            attributes[key]['evidences'] = evidence
    return [attributes[key] for key in sorted(attributes)]


def validate_evidence(product):
    for attribute in product['attributes']:
        for evidence in attribute.get('evidences', []):
            pid, begin, end = (evidence.get(k) for k in ('pid', 'begin', 'end'))
            value = evidence.get('value')
            if any(type(index) is not int for index in (pid, begin, end)):
                raise ValueError('Evidence indices must be integers')
            if not 0 <= pid < len(product['paragraphs']):
                raise ValueError('Evidence paragraph does not exist')
            text = product['paragraphs'][pid]['text']
            if not 0 <= begin < end <= len(text):
                raise ValueError('Evidence character span is out of range')
            if not isinstance(value, str) or text[begin:end] != value:
                raise ValueError('Evidence value differs from cleaned text at its offsets')


def label_mapping(names):
    groups = {}
    for name in sorted(set(names)):
        groups.setdefault(normalize_label(name) or 'ATTRIBUTE', []).append(name)
    mapping = {}
    for base, originals in groups.items():
        for name in originals:
            mapping[name] = base if len(originals) == 1 else base + '_' + hashlib.sha256(name.encode()).hexdigest()[:8].upper()
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Attribute label normalization collision')
    return mapping


def to_bio(product, mapping):
    validate_evidence(product)
    rows, rejected = [], []
    for pid, paragraph in enumerate(product['paragraphs']):
        spans = {}
        for attribute in product['attributes']:
            for evidence in attribute.get('evidences', []):
                if evidence['pid'] == pid:
                    span = {'attribute': mapping[attribute['key']], 'start': evidence['begin'], 'end': evidence['end']}
                    spans[(span['attribute'], span['start'], span['end'])] = span
        selected = []
        for span in sorted(spans.values(), key=lambda s: (-(s['end']-s['start']), s['attribute'], s['start'])):
            if any(span['start'] < other['end'] and other['start'] < span['end'] for other in selected):
                rejected.append(dict(span, product_id=product['id'], pid=pid))
            else:
                selected.append(span)
        selected.sort(key=lambda s: s['start'])
        tokens = tokenize_with_boundaries(paragraph['text'], selected)
        for span in selected:
            covered = [t for t in tokens if span['start'] <= t['start'] and t['end'] <= span['end']]
            if not covered or covered[0]['start'] != span['start'] or covered[-1]['end'] != span['end']:
                raise ValueError('Evidence whitespace boundaries are not exactly representable in BIO')
        rows.append({'id': product['id'] + ':' + str(pid), 'product_id': product['id'],
                     'category': product['category'], 'source': paragraph['source'], 'pid': pid,
                     'text': paragraph['text'], 'tokens': [t['text'] for t in tokens],
                     'token_offsets': [[t['start'], t['end']] for t in tokens],
                     'labels': assign_bio(tokens, selected),
                     'annotated_attributes': [mapping[a['key']] for a in product['attributes']]})
    return rows, rejected


def text_signature(product):
    texts = [(p['source'], ' '.join(p['text'].casefold().split())) for p in product['paragraphs']
             if p['source'] in ('title', 'description', 'feature')]
    return hashlib.sha256(repr(texts).encode()).hexdigest()


def group_products(products, parents=None):
    parents = parents or {}
    union = {p['id']: p['id'] for p in products}
    def find(key):
        while union[key] != key:
            union[key] = union[union[key]]
            key = union[key]
        return key
    def join(a, b):
        a, b = find(a), find(b)
        union[max(a, b)] = min(a, b)
    signatures, families = {}, {}
    for product in sorted(products, key=lambda p: p['id']):
        identity = product['id']
        signature = text_signature(product)
        if signature in signatures:
            join(identity, signatures[signature])
        signatures[signature] = identity
        parent = parents.get(identity)
        if parent:
            if parent in families:
                join(identity, families[parent])
            if parent in union:
                join(identity, parent)
            families[parent] = identity
    return {key: 'MAVE_GROUP_' + hashlib.sha256(find(key).encode()).hexdigest() for key in union}


def split_group(group, seed=42, fractions=None):
    fractions = fractions or {'train': .70, 'val': .15, 'test': .15}
    if set(fractions) != {'train', 'val', 'test'} or any(v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-12:
        raise ValueError('Split fractions must be nonnegative and sum to one')
    value = int(hashlib.sha256(('%d|split|%s' % (seed, group)).encode()).hexdigest(), 16) / 2**256
    if value < fractions['train']:
        return 'train'
    if value < fractions['train'] + fractions['val']:
        return 'val'
    return 'test'
