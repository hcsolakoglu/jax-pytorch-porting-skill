"""Resource-bounded, CPU-only environment for this repository's experiments."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Necessary safety additions: affect only this process and its children.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
if hasattr(os, 'sched_getaffinity'):
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:2])

ROOT = Path(__file__).resolve().parents[1]


def original_module(project: str, filename: str = 'model.py'):
    """Import an allowlisted original without running a training launcher."""
    allowed = {('flax_imagenet', 'models.py'), ('torch_language_model', 'model.py')}
    if (project, filename) not in allowed:
        raise ValueError('Only original model modules are executable through this loader')
    path = ROOT / 'validation' / 'originals' / project / filename
    spec = importlib.util.spec_from_file_location(f'original_{project}', path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
