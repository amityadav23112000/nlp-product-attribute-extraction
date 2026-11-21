"""Run a frozen validation experiment; resumes from atomic product checkpoints."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.llm_extraction.inference import run_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',choices=('wdc','mave'),default='wdc')
    p.add_argument('--data-dir',type=Path,default=Path('datasets/wdc_pave'))
    p.add_argument('--split',choices=('validation',),default='validation')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    metrics,_=run_experiment(json.loads(a.config.read_text()),a.data_dir,a.output_dir,a.dataset,a.smoke)
    print(json.dumps({k:v for k,v in metrics['post_verification'].items() if not isinstance(v,(list,dict))}),flush=True)


if __name__=='__main__':main()
