#!/usr/bin/env python3
"""Bounded, revision-pinned public source retrieval. Retrieval is not review."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.work' / 'sources'
RESEARCH = ROOT / 'research'
MAX_BYTES = 2_000_000
REPOS = {
    'dgrauet/claude-skill-mlx-porting': r'.*',
    'modular/skills': r'.*',
    'mflux-community/mflux': r'^\.cursor/skills/',
    'tensormux/kernel-skills': r'^skills/',
    'Orchestra-Research/AI-Research-SKILLs': r'.*',
    'K-Dense-AI/claude-scientific-skills': r'.*(pytorch|transformers|jax|neural|scikit|numpy|hypothesis|machine-learning|deep-learning|lightning|modal|runpod).*',
    'obra/superpowers': r'^skills/(systematic-debugging|test-driven-development|verification-before-completion|writing-skills)/',
    'apple/coreai-models': r'.*skills/.*',
    'pytorch/pytorch': r'^\.claude/skills/',
    'pytorch/executorch': r'^\.claude/skills/',
    'vllm-project/vllm': r'^\.agents/skills/|^\.claude/skills/',
    'PrathamLearnsToCode/paper2code': r'^skills/',
    'maurinl26/fortranspire': r'^skills/',
    'mindrally/skills': r'.*(machine-learning|deep-learning|pytorch|jax|debug|test).*',
}


def gh_json(endpoint: str) -> object:
    result = subprocess.run(['gh', 'api', endpoint], check=True, capture_output=True,
                            text=True, timeout=40)
    return json.loads(result.stdout)


def retrieve(record: dict) -> dict:
    path = CACHE / record['repository'] / record['path']
    record = dict(record)
    try:
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(record['raw_url'], headers={'User-Agent': 'porting-research/1.0'})
            with urllib.request.urlopen(request, timeout=25) as response:
                content = response.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError('source exceeds 2 MB retrieval limit')
            content.decode('utf-8')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        record.update(status='retrieved_not_reviewed', bytes=len(content),
                      sha256=hashlib.sha256(content).hexdigest(),
                      cache_path=str(path.relative_to(ROOT)))
    except (OSError, ValueError, UnicodeError) as error:
        record.update(status='retrieval_failed', error=f'{type(error).__name__}: {error}')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', action='store_true', help='Retrieve directly adjacent reference text as well as entrypoints')
    parser.add_argument('--repository', action='append', choices=sorted(REPOS))
    parser.add_argument('--shortlist', type=Path, help='JSON list of repository/path pairs to inspect deeply')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(exist_ok=True)
    index_path = ROOT / '.work' / 'repository-index.json'
    old_index = RESEARCH / 'repository-index.json'
    if old_index.exists() and not index_path.exists():
        old_index.replace(index_path)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    shortlist = set(map(tuple, json.loads(args.shortlist.read_text()))) if args.shortlist else None
    records = []
    failures = []
    date = datetime.now(timezone.utc).isoformat()
    for repository in args.repository or REPOS:
        try:
            if repository not in index:
                meta = gh_json(f'repos/{repository}')
                revision = gh_json(f'repos/{repository}/commits/{meta["default_branch"]}')['sha']
                tree = gh_json(f'repos/{repository}/git/trees/{revision}?recursive=1')
                if tree.get('truncated'):
                    raise ValueError('recursive tree truncated; requires scoped traversal')
                paths = [item['path'] for item in tree['tree'] if item['type'] == 'blob']
                index[repository] = {'revision': revision, 'default_branch': meta['default_branch'],
                                     'license': (meta.get('license') or {}).get('spdx_id'),
                                     'accessed': date, 'paths': paths}
                index_path.write_text(json.dumps(index, indent=2) + '\n')
            item = index[repository]
            pattern = re.compile(REPOS[repository], re.I)
            entries = [p for p in item['paths'] if p.endswith('SKILL.md') and pattern.search(p)]
            if shortlist is not None:
                entries = [p for p in entries if (repository, p) in shortlist]
            selected = set(entries)
            if args.support:
                roots = tuple(str(Path(p).parent) + '/' for p in entries)
                selected.update(p for p in item['paths'] if p.startswith(roots) and
                                p.endswith(('.md', '.py', '.yaml', '.json')) and
                                not any(part in p.split('/') for part in ('node_modules', 'vendor', 'fixtures')))
            for path in sorted(selected):
                revision = item['revision']
                records.append({'repository': repository, 'revision': revision, 'path': path,
                                'kind': 'skill' if path.endswith('SKILL.md') else 'supporting_file',
                                'url': f'https://github.com/{repository}/blob/{revision}/{path}',
                                'raw_url': f'https://raw.githubusercontent.com/{repository}/{revision}/{path}',
                                'accessed': date})
            print(f'{repository}: {len(entries)} skill entrypoints, {len(selected)} selected files', flush=True)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failures.append({'repository': repository, 'error': f'{type(error).__name__}: {error}'})
            print(f'{repository}: FAILED {type(error).__name__}', flush=True)
    # Four bounded network workers avoid large clones and unnecessary device load.
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(retrieve, records))
    ledger_path = RESEARCH / 'source-ledger.json'
    previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    merged = {(r['repository'], r['path'], r['revision']): r for r in previous}
    merged.update({(r['repository'], r['path'], r['revision']): r for r in results})
    ledger_path.write_text(json.dumps(list(merged.values()), indent=2) + '\n')
    revisions = {repo: {key: value for key, value in item.items() if key != 'paths'}
                 for repo, item in index.items()}
    (RESEARCH / 'revisions.json').write_text(json.dumps(revisions, indent=2) + '\n')
    (RESEARCH / 'retrieval-failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    print(json.dumps({'selected': len(results), 'retrieved': sum(r['status'] == 'retrieved_not_reviewed' for r in results),
                      'failed_repositories': failures}, indent=2))


if __name__ == '__main__':
    main()
