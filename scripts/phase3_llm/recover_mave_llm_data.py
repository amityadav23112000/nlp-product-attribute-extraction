"""Recover exactly the frozen MAVE pilot, not a new sample or full benchmark."""
from collections import Counter
import argparse,hashlib,json,os
from pathlib import Path
import re,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.mave import official_cleaner,merge_attributes,validate_evidence,metadata_stream
from src.llm_extraction.data_adapter import stratified_sample
from src.llm_extraction.schema_registry import build_registry
from src.llm_extraction.inference import atomic_json
from src.llm_extraction.runtime import require_compute_runtime


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def download(url,path,size,sha):
    if path.exists():
        if path.stat().st_size==size and digest(path)==sha:return
        raise ValueError('Existing source checksum mismatch: '+str(path))
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        print('Download %s; attempt %d; expected %.2f GiB'%(path.name,attempt+1,size/1024**3),flush=True)
        completed=subprocess.run(['curl','--silent','--show-error','--fail','--location','--retry','3',
                                  '--connect-timeout','20','--speed-limit','1024','--speed-time','120',
                                  '--continue-at','-','--output',str(partial),url])
        if completed.returncode==0 and partial.exists() and partial.stat().st_size==size and digest(partial)==sha:
            partial.replace(path);return
    raise RuntimeError('Could not download/verify '+str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('datasets/mave_pilot/llm'))
    parser.add_argument('--source-dir',type=Path,default=Path('/tmp/mave_source'))
    parser.add_argument('--metadata-archive',type=Path,help='Read an already downloaded archive without copying or downloading it')
    args=parser.parse_args();require_compute_runtime()
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    source=args.source_dir;source.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path('datasets/mave_pilot/selection_manifest.json').read_text())
    wanted=set().union(*(set(v) for v in selection['product_ids'].values()))
    if len(wanted)!=5000:raise ValueError('Pilot selection does not contain exactly 5000 products')
    release=json.loads(Path('datasets/mave_pilot/mave_release_manifest.json').read_text())
    labels={};counts=Counter()
    for polarity in ('positive','negative'):
        name='mave_%ss_labels.jsonl'%polarity;spec=release['files'][name];path=source/name
        url='https://media.githubusercontent.com/media/google-research-datasets/MAVE/'+release['revision']+'/labels/'+name
        download(url,path,spec['bytes'],spec['sha256'])
        with path.open('rb') as stream:
            for line in stream:
                match=re.search(rb'"id"\s*:\s*"([^"\\]+)"',line)
                if not match or match[1].decode() not in wanted:continue
                row=json.loads(line);pid=row['id']
                item=labels.setdefault(pid,dict(id=pid,category=row['category'],positive=[],negative=[]))
                if item['category']!=row['category']:raise ValueError('Conflicting release category')
                item[polarity]=row['attributes']
        print('Pilot label identities recovered:',len(labels),flush=True)
    if set(labels)!=wanted:raise ValueError('Pilot labels missing')
    meta=json.loads(Path('datasets/mave_pilot/mave_metadata_manifest.json').read_text())
    archive=args.metadata_archive or source/'All_Amazon_Meta.json.gz'
    if args.metadata_archive:
        if not archive.is_file() or archive.stat().st_size!=meta['bytes'] or digest(archive)!=meta['sha256']:
            raise ValueError('Supplied metadata archive size/checksum does not match the pinned source')
    else:
        download(meta['source_url_without_query'],archive,meta['bytes'],meta['sha256'])
    clean,clean_counts=official_cleaner();products={}
    with metadata_stream(archive) as (stream,layers):
        for line in stream:
            counts['metadata_rows']+=1
            match=re.search(rb'"asin"\s*:\s*"([^"\\]+)"',line)
            if not match:continue
            pid=match[1].decode()
            if pid not in wanted or pid in products:continue
            annotation=labels[pid]
            joined=dict(id=pid,category=annotation['category'],attributes=merge_attributes(annotation['positive'],annotation['negative']))
            product=clean(json.loads(line),joined)
            if product is None:counts['cleaner_rejections']+=1;continue
            validate_evidence(product)
            product['cluster_id']=selection['groups'][pid]
            products[pid]=product
            if len(products)%500==0:print('Pilot products reconstructed:',len(products),flush=True)
    if set(products)!=wanted:raise ValueError('Frozen pilot products missing after metadata scan')
    splits={name:[products[pid] for pid in ids] for name,ids in selection['product_ids'].items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        if {p['cluster_id'] for p in splits[a]}&{p['cluster_id'] for p in splits[b]}:raise ValueError('Pilot group leakage')
    for name,rows in splits.items():
        text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        path=root/(name+'.jsonl')
        if path.exists() and path.read_text()!=text:raise ValueError('Refusing to change recovered pilot split')
        path.write_text(text)
    registry=build_registry(splits['train'],'mave')
    atomic_json(root/'schema_registry.json',registry)
    manifest=dict(dataset='mave',pilot=True,product_counts={n:len(v) for n,v in splits.items()},
                  training_ids=selection['product_ids']['train'],test_ids=selection['product_ids']['test'],
                  selected_validation_ids=[p['id'] for p in stratified_sample(splits['val'],10)],
                  smoke_ids=[p['id'] for p in stratified_sample(splits['val'],2)],seed=42,
                  dataset_sha256={n:digest(root/(n+'.jsonl')) for n in splits},
                  metadata_sha256=meta['sha256'],metadata_gzip_layers=layers,source_revision=release['revision'],
                  selection_sha256=digest(Path('datasets/mave_pilot/selection_manifest.json')))
    atomic_json(root/'manifest.json',manifest)
    atomic_json('results/phase3_llm/mave_recovery.json',dict(status='prepared',manifest=manifest,
                counts=dict(counts),cleaning_counts=dict(clean_counts),note='Exact frozen pilot identities; no new sampling or baseline training'))
    print('MAVE pilot recovered:',manifest['product_counts'],flush=True)


if __name__=='__main__':main()
