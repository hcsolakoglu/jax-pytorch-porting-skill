"""Validate and package only an allowlisted skill directory, reproducibly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT/'skills/jax-pytorch-porting'
MAX_BYTES = 25*1024*1024


def check_local_links(path: Path, boundary: Path) -> int:
    count = 0
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
    for match in re.finditer(r'\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
        link = urlsplit(match.group(1))
        if link.scheme or link.netloc or not link.path:
            continue
        relative = unquote(link.path)
        target = (path.parent/relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(boundary.resolve()) or not target.exists():
            raise ValueError(f'Missing or escaping resource in {path.name}: {relative}')
        count += 1
    return count


def validate_skill(directory: Path) -> list[Path]:
    if directory.is_symlink():
        raise ValueError('Root symlink is not permitted in a distributable package')
    directory = directory.resolve()
    # Inspect links before reading entrypoint or metadata, not after packaging.
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink not permitted in package: {path.relative_to(directory)}')
    entry = directory/'SKILL.md'
    text = entry.read_text(encoding='utf-8')
    if not text.startswith('---\n') or '\n---\n' not in text[4:]:
        raise ValueError('Missing YAML frontmatter')
    metadata = yaml.safe_load(text.split('---', 2)[1])
    if not isinstance(metadata, dict) or set(metadata) != {'name', 'description'}:
        raise ValueError('Portable core requires exactly name and description')
    name, description = metadata['name'], metadata['description']
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) or len(name) > 64 or name != directory.name:
        raise ValueError('Invalid skill name or directory mismatch')
    if not isinstance(description, str) or not 1 <= len(description) <= 1024:
        raise ValueError('Description must contain 1-1024 characters')
    if len(text.splitlines()) > 500:
        raise ValueError('Move deep references out of the core')
    ui = yaml.safe_load((directory/'agents/openai.yaml').read_text())
    if not all(ui.get('interface', {}).get(key) for key in ('display_name', 'short_description')):
        raise ValueError('Missing UI metadata')
    allowed_roots = {'SKILL.md', 'LICENSE', 'agents', 'references', 'scripts'}
    files = []
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if path.is_symlink():
            raise ValueError(f'Symlink not permitted in package: {relative}')
        if '__pycache__' in relative.parts or path.suffix == '.pyc':
            continue
        if relative.parts[0] not in allowed_roots:
            raise ValueError(f'Unexpected package content: {relative}')
        if path.is_file():
            if path.name != 'LICENSE' and path.suffix not in {'.md', '.py', '.yaml'}:
                raise ValueError(f'Unexpected file type: {relative}')
            files.append(path)
    if not (directory/'LICENSE').is_file():
        raise ValueError('Missing package license')
    if sum(path.stat().st_size for path in files) > MAX_BYTES:
        raise ValueError('Uncompressed package exceeds 25 MiB')
    for path in files:
        if path.suffix == '.md':
            check_local_links(path, directory)
    return files


def build_package(directory: Path, destination: Path) -> dict:
    files = validate_skill(directory)
    directory = directory.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.skill-', suffix='.zip', dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                info = zipfile.ZipInfo(str(Path(directory.name)/path.relative_to(directory)), (2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        if temporary.stat().st_size > MAX_BYTES:
            raise ValueError('Compressed package exceeds 25 MiB')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive': destination.name, 'bytes': destination.stat().st_size,
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'files': {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            'core_words': len((directory/'SKILL.md').read_text().split()),
            'scope': 'Skill runtime guidance only; no model ports, third-party snapshots, dependencies or experiment data'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'dist/skill.zip')
    args = parser.parse_args()
    report = build_package(SKILL, args.output)
    (args.output.parent/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
