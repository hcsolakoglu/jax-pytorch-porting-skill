#!/usr/bin/env python3
"""Fetch only pinned original model sources, never target-framework ports."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "flax_imagenet": (
        "google/flax",
        "01854da11286b4109c59d7fd9205f3822fe807d6",
        ["examples/imagenet/models.py", "examples/imagenet/train.py", "LICENSE"],
    ),
    "torch_language_model": (
        "pytorch/examples",
        "acc295dc7b90714f1bf47f06004fc19a7fe235c4",
        [
            "word_language_model/model.py",
            "word_language_model/main.py",
            "word_language_model/generate.py",
            "LICENSE",
        ],
    ),
}


def main() -> None:
    manifest = []
    for name, (repository, revision, paths) in SOURCES.items():
        destination = ROOT / "validation" / "originals" / name
        destination.mkdir(parents=True, exist_ok=True)
        for source_path in paths:
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{source_path}"
            with urllib.request.urlopen(url, timeout=20) as response:
                content = response.read(500_001)
            if len(content) > 500_000:
                raise ValueError(f"Oversized original source: {source_path}")
            content.decode("utf-8")
            local_path = destination / Path(source_path).name
            local_path.write_bytes(content)
            manifest.append(
                {
                    "repository": repository,
                    "revision": revision,
                    "source_path": source_path,
                    "url": url,
                    "local_path": str(local_path.relative_to(ROOT)),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            print(f"{name}/{local_path.name}: {len(content)} bytes", flush=True)
    (ROOT / "validation" / "originals" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
