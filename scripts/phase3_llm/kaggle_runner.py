"""Portable entry points used by the Kaggle notebooks; no scheduler or training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
MODES=('zero_shot','schema_guided','one_shot','three_shot')


def configure():
    os.chdir(ROOT)
    if os.environ.get('LLM_EXECUTION_BACKEND')!='local':  # run_local.py keeps its own backend/cache
        os.environ['LLM_EXECUTION_BACKEND']='kaggle'
        os.environ['HF_HOME']=str(Path(tempfile.gettempdir())/'qwen_kaggle_cache')
        os.environ['HF_HUB_CACHE']=str(Path(os.environ['HF_HOME'])/'hub')
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['USE_TF']='0'
    os.environ['USE_FLAX']='0'
    os.environ['PYTHONUNBUFFERED']='1'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive,destination):
    """Reject traversal, absolute members and symlinks before extracting."""
    destination=Path(destination).resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target=(destination/member.filename).resolve()
            if not target.is_relative_to(destination) or ((member.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe archive member: '+member.filename)
        stream.extractall(destination)


def verify_bundle():
    manifest=json.loads((ROOT/'configs/llm_integrity_manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        path=ROOT/relative
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Bundle integrity failure: '+relative)
    print('Verified bundle:',len(manifest['files']),'files; base commit',manifest['base_git_commit'])
    return manifest


def preflight_gpu():
    configure()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator in Kaggle Notebook settings before continuing.')
    # Exercise a kernel before downloading several gigabytes of weights.
    try:
        value=(torch.ones(1,device='cuda')+1).item()
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError('The installed PyTorch cannot execute on this accelerator. Choose a compatible Kaggle GPU/image; no model has been downloaded yet.') from error
    free,total=torch.cuda.mem_get_info()
    devices=[dict(index=i,name=torch.cuda.get_device_name(i),free_gib=round(torch.cuda.mem_get_info(i)[0]/1024**3,2))
             for i in range(torch.cuda.device_count())]
    info=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
              bf16=torch.cuda.is_bf16_supported(),free_gib=round(free/1024**3,2),total_gib=round(total/1024**3,2),tensor_check=value,devices=devices)
    print(json.dumps(info,indent=2))
    sufficient=free>=10*1024**3 if len(devices)==1 else min(d['free_gib'] for d in devices)>=6 and sum(d['free_gib'] for d in devices)>=14
    if not sufficient:
        raise RuntimeError('The full-precision 4B experiment needs more free GPU memory. Use an accelerator with at least roughly 12–16 GiB available; quantization is not silently enabled.')
    from importlib.metadata import version
    profile=dict(gpus=[d['name'] for d in devices],dtype='bfloat16' if info['bf16'] else 'float16',
                 versions={p:version(p) for p in ('torch','transformers','accelerate')})
    os.environ['KAGGLE_RUNTIME_PROFILE']=json.dumps(profile,sort_keys=True)
    (ROOT/'KAGGLE_GPU.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def download_model():
    configure();preflight_gpu()
    os.environ['HF_HUB_OFFLINE']='0';os.environ['TRANSFORMERS_OFFLINE']='0'
    from huggingface_hub import snapshot_download
    config=json.loads((ROOT/'configs/llm_zero_shot.json').read_text())
    print('Downloading pinned model:',config['model'],config['revision'])
    path=snapshot_download(config['model'],revision=config['revision'],
             cache_dir=os.environ['HF_HUB_CACHE'],max_workers=2,
             allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE'])
    print('Model cached outside notebook outputs:',path)
    return path


def paths(dataset):
    if dataset not in ('wdc','mave'):raise ValueError('Dataset must be wdc or mave')
    data=ROOT/('datasets/wdc_pave' if dataset=='wdc' else 'datasets/mave_pilot/llm')
    return data,ROOT/'results/phase3_llm'/(dataset+'_validation'),ROOT/'results/phase3_llm'/(dataset+'_test')


def check_data(dataset):
    data,_,_=paths(dataset)
    if not (data/'manifest.json').exists():
        raise FileNotFoundError('Prepared '+dataset+' data is missing. For MAVE run the CPU preparation notebook, attach its prepared ZIP, and call import_mave_data().')
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['dataset']!=dataset:raise ValueError('Prepared dataset identity differs')
    for split,expected in manifest['dataset_sha256'].items():
        if sha(data/(split+'.jsonl'))!=expected:raise ValueError('Prepared split changed: '+split)
    partitions={}
    for split in ('train','val','test'):
        with (data/(split+'.jsonl')).open() as stream:
            identities=[str(json.loads(line)['id']) for line in stream if line.strip()]
        if len(identities)!=len(set(identities)):raise ValueError('Duplicate product IDs in '+split)
        partitions[split]=set(identities)
    if partitions['train']!=set(manifest['training_ids']) or partitions['test']!=set(manifest['test_ids']):
        raise ValueError('Prepared product IDs differ from manifest')
    if not set(manifest['selected_validation_ids'])<=partitions['val'] or not set(manifest['smoke_ids'])<=partitions['val']:
        raise ValueError('Validation sample is outside the validation split')
    if any(partitions[a]&partitions[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Product leakage across dataset splits')
    expected=354 if dataset=='wdc' else 716
    if len(manifest['test_ids'])!=expected:raise ValueError('Test coverage differs from frozen policy')
    print(dataset,':',len(manifest['training_ids']),'train,',len(manifest['selected_validation_ids']),
          'comparison,',len(manifest['smoke_ids']),'smoke,',len(manifest['test_ids']),'test products')
    return manifest


def run(dataset,stage):
    configure();check_data(dataset)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    data,validation,test=paths(dataset)
    if stage=='smoke':
        command=['scripts/phase3_llm/run_llm_extraction.py','--config','configs/llm_zero_shot.json',
                 '--dataset',dataset,'--data-dir',str(data),'--output-dir',str(validation/'smoke'),'--smoke']
    elif stage=='validation':
        command=['scripts/phase3_llm/run_llm_ablation.py','--dataset',dataset,'--data-dir',str(data),'--output-dir',str(validation)]
    elif stage=='test':
        command=['scripts/phase3_llm/run_llm_test.py','--dataset',dataset,'--data-dir',str(data),
                 '--validation-dir',str(validation),'--output-dir',str(test)]
    else:raise ValueError('Stage must be smoke, validation or test')
    print('RUN:',dataset,stage,flush=True)
    # Always retain a downloadable checkpoint ZIP, including if a process fails.
    try:
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=os.environ.copy(),check=True)
    finally:
        export_results(dataset)


def export_results(dataset):
    _,validation,test=paths(dataset)
    destination=ROOT.parent/(dataset+'_qwen_results.zip')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as stream:
        for folder in (validation,test):
            if folder.exists():
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and path.suffix!='.tmp':stream.write(path,path.relative_to(ROOT))
        for name in ('KAGGLE_GPU.json','BUNDLE_MANIFEST.json'):
            if (ROOT/name).exists():stream.write(ROOT/name,name)
    print('Download/save this results checkpoint:',destination,flush=True)
    return destination


def restore_results(archive):
    """Restore only saved results, never overwrite packaged source or data."""
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            if member.is_dir():continue
            if member.filename in ('KAGGLE_GPU.json','BUNDLE_MANIFEST.json'):continue
            relative=Path(member.filename)
            allowed=any(relative.is_relative_to(Path('results/phase3_llm')/(d+'_'+s))
                        for d in ('wdc','mave') for s in ('validation','test'))
            target=(ROOT/relative).resolve()
            if not allowed or not target.is_relative_to(ROOT):raise ValueError('Unexpected results member: '+member.filename)
            data=stream.read(member)
            if target.exists() and target.read_bytes()!=data:raise ValueError('Existing result differs; restore into a fresh workspace: '+str(relative))
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print('Results restored. Rerun the same stage to resume unchanged product checkpoints.')


def prepare_mave(metadata_archive=None):
    configure()
    source=Path(tempfile.gettempdir())/'mave_pilot_sources'
    source.mkdir(parents=True,exist_ok=True)
    if metadata_archive is None:
        specs=[json.loads((ROOT/'datasets/mave_pilot/mave_metadata_manifest.json').read_text())['bytes']]
        release=json.loads((ROOT/'datasets/mave_pilot/mave_release_manifest.json').read_text())
        specs += [release['files'][n]['bytes'] for n in ('mave_positives_labels.jsonl','mave_negatives_labels.jsonl')]
        existing=sum(p.stat().st_size for p in source.iterdir() if p.is_file())
        required=max(0,sum(specs)-existing)+1024**3
        if shutil.disk_usage(source).free<required:
            raise RuntimeError('Insufficient temporary disk for metadata recovery. Attach the pinned Amazon archive and pass metadata_archive=..., or attach already prepared pilot data.')
    command=[sys.executable,'-u','scripts/phase3_llm/recover_mave_llm_data.py','--source-dir',str(source)]
    if metadata_archive is not None:command += ['--metadata-archive',str(metadata_archive)]
    subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True)
    manifest=check_data('mave')
    if manifest['selection_sha256']!=sha(ROOT/'datasets/mave_pilot/selection_manifest.json'):
        raise ValueError('Recovered pilot identities differ from frozen experiment')
    data,_,_=paths('mave');destination=ROOT.parent/'mave_pilot_prepared.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(data.iterdir()):
            if path.is_file():stream.write(path,Path('mave_llm')/path.name)
        report=ROOT/'results/phase3_llm/mave_recovery.json'
        stream.write(report,'provenance/mave_recovery.json')
    print('Prepared pilot only; save and attach to the GPU notebook:',destination)
    print('Large metadata sources remain in temporary storage and are not exported.')
    return destination


def import_mave_data(archive=None):
    configure();target,_,_=paths('mave')
    if (target/'manifest.json').exists():return check_data('mave')
    if archive is None:
        zipped=sorted(Path('/kaggle/input').rglob('mave_pilot_prepared.zip'))
        if len(zipped)>1:raise ValueError('Multiple prepared ZIPs; pass the intended archive explicitly')
        if zipped:archive=zipped[0]
    if archive is not None:
        with tempfile.TemporaryDirectory() as directory:
            safe_extract(archive,directory)
            folder=Path(directory)/'mave_llm'
            if not (folder/'manifest.json').is_file():raise ValueError('Prepared archive lacks mave_llm/manifest.json')
            shutil.copytree(folder,target)
    else:
        candidates=[]
        for path in Path('/kaggle/input').rglob('manifest.json'):
            if path.parent.name=='mave_llm':candidates.append(path.parent)
        if len(candidates)!=1:raise FileNotFoundError('Attach mave_pilot_prepared.zip (or its extracted mave_llm folder) from the CPU preparation notebook.')
        shutil.copytree(candidates[0],target)
    manifest=check_data('mave')
    if manifest['selection_sha256']!=sha(ROOT/'datasets/mave_pilot/selection_manifest.json'):
        raise ValueError('Uploaded pilot has different frozen product IDs')
    return manifest
