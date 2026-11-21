"""Explicit compute backends; never fabricate a SLURM allocation in notebooks.

slurm  - HPC GPU node (original Phase 2/3 environment)
kaggle - Kaggle notebook GPU (T4 x2 / P100)
local  - a local machine: CUDA if present, otherwise Apple-Silicon MPS (e.g. MacBook M-series)
"""
import os
from pathlib import Path


def require_compute_runtime():
    backend=os.environ.get('LLM_EXECUTION_BACKEND','slurm')
    if backend=='slurm':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('An allocated SLURM compute node is required')
    elif backend=='kaggle':
        if not Path('/kaggle/working').is_dir():
            raise RuntimeError('Kaggle backend requires /kaggle/working')
    elif backend!='local':
        raise ValueError('LLM_EXECUTION_BACKEND must be slurm, kaggle or local')
    return backend


def accelerator(torch):
    if torch.cuda.is_available():return 'cuda'
    if getattr(torch.backends,'mps',None) is not None and torch.backends.mps.is_available():return 'mps'
    raise RuntimeError('No CUDA or Apple MPS accelerator available')
