"""Bind numerical evidence to exact local inputs, code and dependency lock."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def fingerprint() -> dict:
    paths = [ROOT/'validation/budgets.json', ROOT/'research/validation-environment.lock',
             ROOT/'skills/jax-pytorch-porting/scripts/parity.py']
    paths += [path for path in (ROOT/'validation').rglob('*.py')
              if path.name != 'benchmark_cpu.py' and '__pycache__' not in path.parts]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'validation_sha256': digest, 'files': hashes}


def provenance() -> dict:
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    return {**fingerprint(), 'base_git_commit': revision, 'python': platform.python_version(),
            'architecture': platform.machine(), 'versions': {
                name: importlib.metadata.version(name) for name in ('torch', 'jax', 'jaxlib', 'flax', 'optax', 'numpy')}}
