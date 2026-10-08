from pathlib import Path
import zipfile

import pytest

from tools.package_skill import build_package, validate_skill


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "example"
    (root / "agents").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: example\ndescription: Test a bounded example workflow.\n---\n# Example\n"
    )
    (root / "agents/openai.yaml").write_text(
        "interface:\n  display_name: Example\n  short_description: A test skill\n"
    )
    (root / "LICENSE").write_text("Test license\n")
    return root


def test_package_is_repeatable_and_excludes_cache(skill, tmp_path):
    (skill / "__pycache__").mkdir()
    (skill / "__pycache__/temporary.pyc").write_bytes(b"cache")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    assert build_package(skill, first)["sha256"] == build_package(skill, second)["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert len(archive.namelist()) == 3
        assert all(not Path(name).is_absolute() for name in archive.namelist())


def test_package_rejects_root_symlink(skill, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(skill, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_skill(alias)


@pytest.mark.parametrize("link", ["references/missing.md", "../outside.md"])
def test_package_rejects_missing_or_escaping_reference(skill, link):
    (skill.parent / "outside.md").write_text("Must not read or package this file")
    with (skill / "SKILL.md").open("a") as stream:
        stream.write(f"\n[Required reference]({link})\n")
    with pytest.raises(ValueError):
        validate_skill(skill)


def test_package_rejects_unexpected_data(skill):
    (skill / "weights.bin").write_bytes(b"not a skill resource")
    with pytest.raises(ValueError):
        validate_skill(skill)
