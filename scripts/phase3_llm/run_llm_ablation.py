"""Smoke test then controlled, same-product validation prompt ablations."""
import argparse,json
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.llm_extraction.inference import run_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dataset',choices=('wdc','mave'),default='wdc')
    p.add_argument('--data-dir',type=Path,default=Path('datasets/wdc_pave'))
    a=p.parse_args();extractor=None
    first=json.loads(Path('configs/llm_zero_shot.json').read_text())
    smoke,extractor=run_experiment(first,a.data_dir,a.output_dir/'smoke',dataset=a.dataset,smoke=True)
    if smoke['reliability']['generation_failures'] or smoke['reliability']['json_parse_rate']<0.8:
        raise RuntimeError('Smoke acceptance failed; inspect diagnostics before large inference')
    for name in ('zero_shot','schema_guided','one_shot','three_shot'):
        config=json.loads(Path('configs/llm_'+name+'.json').read_text())
        metrics,extractor=run_experiment(config,a.data_dir,a.output_dir/name,dataset=a.dataset,extractor=extractor)
    subprocess.run([sys.executable,'scripts/phase3_llm/generate_llm_report.py','--run-dir',str(a.output_dir),
                    '--data-dir',str(a.data_dir),'--dataset',a.dataset],check=True)


if __name__=='__main__':main()
