"""Test Vercel installation in an ephemeral HOME; never modify actual user skills."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.5.25'


def main():
    (ROOT/'.work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='install-smoke-', dir=ROOT/'.work') as directory:
        temporary = Path(directory)
        home = temporary/'home'
        home.mkdir()
        environment = os.environ.copy()
        environment.update(HOME=str(home), XDG_CONFIG_HOME=str(home/'.config'),
                           CODEX_HOME=str(home/'.codex'), CLAUDE_CONFIG_DIR=str(home/'.claude'),
                           npm_config_cache=str(temporary/'npm-cache'), DO_NOT_TRACK='1', CI='1')
        command = ['timeout', '--signal=TERM', '--kill-after=5s', '65s', 'npx', '--yes', f'skills@{VERSION}',
                   'add', str(ROOT/'skills/jax-pytorch-porting'), '--skill', 'jax-pytorch-porting',
                   '--global', '--agent', 'codex', 'claude-code', 'cursor', 'antigravity', '--yes']
        result = subprocess.run(command, cwd=temporary, env=environment, capture_output=True,
                                stdin=subprocess.DEVNULL, text=True, timeout=75)
        entry_hash = hashlib.sha256((ROOT/'skills/jax-pytorch-porting/SKILL.md').read_bytes()).hexdigest()
        paths = ['.agents/skills/jax-pytorch-porting', '.claude/skills/jax-pytorch-porting',
                 '.cursor/skills/jax-pytorch-porting', '.codex/skills/jax-pytorch-porting',
                 '.gemini/antigravity/skills/jax-pytorch-porting', '.gemini/config/skills/jax-pytorch-porting']
        inspected = []
        for relative in paths:
            path = home/relative
            entry = path/'SKILL.md'
            inspected.append({'path': '~/' + relative, 'exists': entry.is_file(), 'symlink': path.is_symlink(),
                              'entry_matches': entry.is_file() and hashlib.sha256(entry.read_bytes()).hexdigest() == entry_hash})
        report = {'evidence_class': 'OBSERVED', 'installer': f'skills@{VERSION}', 'exit_code': result.returncode,
                  'scope': 'Ephemeral HOME installation paths and file integrity only; no agent model invoked',
                  'paths': inspected, 'actual_user_home_modified': False,
                  'prior_attempt': 'Inherited-stdin run timed out after startup with stopped processes; retry closes stdin.',
                  'output_tail': (result.stdout+result.stderr)[-3500:].replace(str(temporary), '<temporary>').replace(str(ROOT), '<project>')}
        (ROOT/'research/installation-smoke.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'output_tail'}, indent=2))
        if result.returncode != 0:
            raise RuntimeError('Installer failed; see sanitized report')
        if not inspected[0]['entry_matches'] or not inspected[1]['entry_matches']:
            raise RuntimeError('Canonical or Claude skill content does not match')


if __name__ == '__main__':
    main()
