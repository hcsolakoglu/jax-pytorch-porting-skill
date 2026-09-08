#!/usr/bin/env python3
"""Retrieve a bounded, predeclared matched cohort of released refinement traces."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import math
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'ScalingIntelligence/kernelbench-samples'
REVISION = 'b32b70bf2d84c7395a94a214e83183152233f298'
MODELS = ('deepseek-R1', 'deepseek-v3', 'llama-3.1-70b-inst')
CASES = ((1, 1), (1, 19), (1, 40), (2, 37))
MAX_BYTES = 3_000_000


def timing_eligible(evaluation: dict) -> bool:
    runtime = evaluation.get('runtime')
    metadata = evaluation.get('metadata') or {}
    return (evaluation.get('correctness') is True
            and isinstance(runtime, (int, float)) and not isinstance(runtime, bool)
            and math.isfinite(runtime) and runtime > 0
            and not any('error' in key.lower() for key in metadata))


def retrieve(item):
    level, problem, model = item
    remote = f'iterative_refinement/level{level}/eval_result_last_only/{model}/problem_{problem}/sample_0/log.json'
    url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote}'
    path = ROOT / '.work/traces/logs' / f'level{level}-problem{problem}-{model}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(MAX_BYTES+1)
        if len(content) > MAX_BYTES:
            raise ValueError('Trace exceeds bounded download size')
        path.write_bytes(content)
    payload = json.loads(content)
    rounds = []
    previous = ''
    for key in sorted((key for key in payload if key.isdigit()), key=int):
        record = payload[key]
        code = record.get('kernel_code') or ''
        evaluation = record.get('eval_result') or {}
        diff = list(difflib.unified_diff(previous.splitlines(), code.splitlines(), n=1))
        rounds.append({'turn': int(key), 'compiled': evaluation.get('compiled'),
                       'correctness': evaluation.get('correctness'), 'runtime_ms': evaluation.get('runtime'),
                       'timing_eligible': timing_eligible(evaluation),
                       'metadata': evaluation.get('metadata'), 'runtime_stats': evaluation.get('runtime_stats'),
                       'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                       'code_lines': len(code.splitlines()), 'changed_diff_lines': len(diff),
                       'explicit_feedback_field_bytes': len(record.get('feedback', '')),
                       'context_bytes': len(record.get('context', ''))})
        previous = code
    return {'model': model, 'level': level, 'problem_id': problem, 'metadata': payload.get('metadata'),
            'url': url, 'revision': REVISION, 'sha256': hashlib.sha256(content).hexdigest(),
            'bytes': len(content), 'cache_path': str(path.relative_to(ROOT)),
            'rounds': rounds,
            'interpretation': 'Published results, not rerun. Missing feedback field is not evidence of no feedback; context may contain it. Compiled=false can include runtime/environment errors.'}


def main():
    cohort = [(level, problem, model) for level, problem in CASES for model in MODELS]
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(retrieve, cohort))
    output = ROOT / 'research/trace-cohort.json'
    output.write_text(json.dumps(records, indent=2) + '\n')
    for record in records:
        states = ''.join('P' if row['correctness'] else 'C' if row['compiled'] else 'F' for row in record['rounds'])
        times = [row['runtime_ms'] for row in record['rounds'] if row['timing_eligible']]
        print(f"L{record['level']} P{record['problem_id']:02} {record['model']}: {states} passing_runtime_ms={times}")
    print(f'{len(records)} complete released traces, {sum(len(r["rounds"]) for r in records)} attempts; no generated code executed')


if __name__ == '__main__':
    main()
