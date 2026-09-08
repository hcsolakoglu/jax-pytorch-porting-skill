"""Cheap offline release gates; no framework imports or network execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Allow direct CLI invocation and package imports without installing this repo.
sys.path.insert(0, str(ROOT))
from tools.package_skill import SKILL, check_local_links, validate_skill  # noqa: E402
from tools.render_comparison import render, validated_entries  # noqa: E402
from validation.evidence import fingerprint  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    packaged = validate_skill(SKILL)
    process = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                             cwd=ROOT, check=True, capture_output=True, timeout=10)
    paths = sorted({ROOT/name for name in process.stdout.decode().split('\0') if name})
    local_links = python_files = 0
    forbidden_parts = {'.venv', '.work', '__pycache__', '.pytest_cache', '.ruff_cache', 'dist'}
    secret_patterns = (r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{50,}',
                       r'sk-proj-[A-Za-z0-9_-]{40,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for path in paths:
        relative = path.relative_to(ROOT)
        require(not forbidden_parts.intersection(relative.parts), f'Forbidden tracked artifact: {relative}')
        require(not path.is_symlink(), f'Unexpected repository symlink: {relative}')
        require(path.is_file() and path.stat().st_size <= 2_000_000, f'Unexpected large/non-file artifact: {relative}')
        text = path.read_text(encoding='utf-8')
        require(not any(re.search(pattern, text) for pattern in secret_patterns), f'Possible secret in {relative}')
        if path.suffix == '.md':
            local_links += check_local_links(path, ROOT)
        if path.suffix == '.py':
            ast.parse(text, filename=str(relative))
            python_files += 1
    originals = json.loads((ROOT/'validation/originals/manifest.json').read_text())
    for record in originals:
        path = ROOT/record['local_path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], f'Original source changed: {path.name}')
    comparison = json.loads((ROOT/'research/comparison-input.json').read_text())
    validated_entries(comparison)
    require((ROOT/'research/competitors.md').read_text() == render(), 'Comparison document is stale')
    iterations = [int(value) for value in re.findall(r'^## Iteration (\d+):',
                                                   (ROOT/'research/iterations.md').read_text(), flags=re.M)]
    require(iterations == list(range(1, len(iterations)+1)) and len(iterations) >= 10,
            'At least ten ordered substantive iteration records required')
    current = fingerprint()['validation_sha256']
    reports = ['resnet-forward.json', 'resnet-training.json', 'rnn-training.json']
    for name in reports:
        report = json.loads((ROOT/'validation/results'/name).read_text())
        require(report['status'] == 'PASS', f'Failed model evidence: {name}')
        require(report.get('provenance', {}).get('validation_sha256') == current, f'Stale model evidence: {name}')
    for name in ['benchmark-resnet.json', 'benchmark-rnn.json', 'benchmark-resnet-inductor.json']:
        report = json.loads((ROOT/'validation/results'/name).read_text())
        require(report.get('provenance', {}).get('validation_sha256') == current, f'Stale benchmark prerequisite: {name}')
        require(report['benchmark_sha256'] == hashlib.sha256((ROOT/'validation/benchmark_cpu.py').read_bytes()).hexdigest(),
                f'Stale benchmark implementation: {name}')
    result = {'status': 'PASS', 'scope': 'Offline structure, links, syntax, provenance and packaging prerequisites',
              'repository_files': len(paths), 'python_files_parsed': python_files, 'local_links_checked': local_links,
              'original_hashes_verified': len(originals), 'packaged_files': len(packaged),
              'competitors': 25, 'individual_scores': 500, 'iterations': len(iterations),
              'model_evidence_fresh': True,
              'performance_gate': 'ResNet CPU no-regression remains FAILED; not concealed by release checks',
              'remote_links': 'Selected authoritative sources reviewed separately; no claim of exhaustive live HTTP checks'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
