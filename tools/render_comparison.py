"""Render manually justified scores; never infer scores from keyword counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validated_entries(data: dict) -> list[dict]:
    entries = data["entries"]
    if len(entries) != 25 or len({(row["repository"], row["path"]) for row in entries}) != 25:
        raise ValueError("Exactly 25 unique competitors are required")
    for row in entries + [data["ours"]]:
        if len(row["scores"]) != 20 or any(
            type(score) is not int or not 0 <= score <= 5 for score in row["scores"]
        ):
            raise ValueError("Each row requires twenty integer scores from zero to five")
    for row in entries:
        if not all(row.get(key) for key in ("strength", "weakness", "evidence")):
            raise ValueError("Every competitor needs explicit pros, cons and evidence")
    return sorted(
        entries,
        key=lambda row: (
            -sum(row["scores"]),
            -row["scores"][2],
            -row["scores"][5],
            -row["scores"][4],
            row["repository"] + "/" + row["path"],
        ),
    )


def render() -> str:
    data = json.loads((ROOT / "research/comparison-input.json").read_text())
    rows = validated_entries(data)
    ledger = {
        (row["repository"], row["path"]): row
        for row in json.loads((ROOT / "research/source-ledger.json").read_text())
    }
    gist = json.loads((ROOT / "research/equinox-gist.json").read_text())
    headings = [
        "JAX",
        "PyTorch",
        "Bidirectional",
        "Inference",
        "Training",
        "Numerics",
        "Gradients",
        "Optimizer",
        "Performance",
        "Accelerators",
        "Localization",
        "Recovery",
        "Fast fail",
        "Adversarial",
        "Regression",
        "Harness",
        "Context",
        "Documentation",
        "Usability",
        "Evidence",
    ]
    text = [
        "# Related skills: 25-by-20 evidence-based comparison",
        "",
        f"Review date: {data['review_date']}. [Frozen rubric](rubric.md); [manual scores](comparison-input.json).",
        "",
        data["scope"],
        "",
        data["selection"],
        "",
        data["uncertainty"],
        "",
        "**Scores measure documented fitness for this task, not general quality or measured agent success.**",
        "",
        "## Ranking",
        "",
        "| Rank | Related skill | Score / 100 |",
        "|---:|---|---:|",
    ]
    for index, row in enumerate(rows, 1):
        label = row["path"].split("/")[-2] if "/" in row["path"] else "pytorch-to-equinox"
        text.append(f"| {index} | {row['repository']}: `{label}` | {sum(row['scores'])} |")
    own = data["ours"]
    own_rank = 1 + sum(sum(row["scores"]) > sum(own["scores"]) for row in rows)
    text += [
        "",
        f"**Our self-assessed score: {sum(own['scores'])} / 100, position {own_rank} when added to this inspected set.**",
        "This is not a blinded assessment, global skill ranking or controlled proof that an agent will perform best.",
        own["limitations"],
        "",
        "## Complete criterion matrix",
        "",
    ]
    for start in (0, 10):
        text += [
            f"### C{start + 1:02d} through C{start + 10:02d}",
            "",
            "| Entry | " + " | ".join(f"C{index + 1:02d}" for index in range(start, start + 10)) + " |",
            "|---|" + "---:|" * 10,
        ]
        for index, row in enumerate(rows, 1):
            text.append(f"| {index} | " + " | ".join(map(str, row["scores"][start : start + 10])) + " |")
        text.append("| Ours | " + " | ".join(map(str, own["scores"][start : start + 10])) + " |")
        text += [
            "",
            "; ".join(f"C{index + 1:02d} {headings[index]}" for index in range(start, start + 10)),
            "",
        ]
    text += ["## Per-skill evidence, strengths and weaknesses", ""]
    for index, row in enumerate(rows, 1):
        if row["repository"] == "nboyd/pytorch-to-equinox":
            source = {
                "url": gist["url"] + "/" + gist["revision"],
                "revision": gist["revision"],
                "sha256": gist["sha256"],
            }
        else:
            source = ledger[(row["repository"], row["path"])]
        text += [
            f"### {index}. {row['repository']} / {row['path']}",
            "",
            f"Score: **{sum(row['scores'])} / 100**. [Pinned source]({source['url']}).",
            f"Revision `{source['revision']}`; entry SHA256 `{source['sha256']}`.",
            "",
            "**Strengths:** " + row["strength"],
            "",
            "**Limitations:** " + row["weakness"],
            "",
            "**Inspected evidence:** " + row["evidence"],
            "",
        ]
    text += ["## Our score evidence", "", "| Criterion | Score | Evidence / limitation |", "|---|---:|---|"]
    for index, (score, evidence) in enumerate(zip(own["scores"], own["evidence"], strict=True)):
        text.append(f"| C{index + 1:02d} {headings[index]} | {score} | {evidence} |")
    text += [
        "",
        "## Interpretation and missing evaluation",
        "",
        "A high aggregate favors broad bidirectional coverage. A CUDA microbenchmark or compiler-debugging specialist can be stronger in its own domain despite a lower total. Rows were scored against their inspected skill scope, not parent-repository popularity, stars or unrelated tooling.",
        "",
        "No matched agent trials were run on these 25 skills. A defensible success-rate ranking would require equal models, budgets, tasks, hidden test sets and independent judges. Small ordinal score differences are not meaningful confidence intervals. Our negative ResNet performance result remains a failed development performance gate, regardless of this documentation score.",
        "",
        "Reproduce the table with `python tools/render_comparison.py`. The renderer validates totals and ordering; it does not validate subjective judgments. A new rubric version requires rescoring every row.",
    ]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "research/competitors.md"
    content = render()
    if args.check:
        if not output.is_file() or output.read_text() != content:
            raise SystemExit("Comparison is missing or stale")
    else:
        output.write_text(content)
    data = json.loads((ROOT / "research/comparison-input.json").read_text())
    rows = validated_entries(data)
    print(
        json.dumps(
            {
                "competitors": len(rows),
                "criterion_scores": 20 * len(rows),
                "own_total": sum(data["ours"]["scores"]),
                "top_related_total": sum(rows[0]["scores"]),
            }
        )
    )


if __name__ == "__main__":
    main()
