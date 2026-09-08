#!/usr/bin/env python3
"""Summarize released trace records without trusting their boolean success flag."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def valid_performance(row: dict) -> bool:
    value = row.get('runtime_ms')
    return (row.get('timing_eligible') is True and row.get('compiled') is True and row.get('correctness') is True
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0
            and bool(row.get('runtime_stats')))


def main() -> None:
    traces = json.loads((ROOT / 'research/trace-cohort.json').read_text())
    output = []
    for trace in traces:
        rounds = trace['rounds']
        valid = [row for row in rounds if valid_performance(row)]
        duplicates = sum(left['code_sha256'] == right['code_sha256']
                         for left, right in zip(rounds, rounds[1:]))
        best = min(valid, key=lambda row: row['runtime_ms']) if valid else None
        output.append({'model': trace['model'], 'level': trace['level'], 'problem_id': trace['problem_id'],
                       'url': trace['url'], 'attempts': len(rounds),
                       'reported_correct': sum(row['correctness'] is True for row in rounds),
                       'valid_timed_attempts': len(valid), 'consecutive_unchanged_code': duplicates,
                       'best_valid_turn': best['turn'] if best else None,
                       'best_reported_runtime_ms': best['runtime_ms'] if best else None,
                       'last_turn_valid': valid_performance(rounds[-1]),
                       'last_turn_runtime_ms': rounds[-1]['runtime_ms'],
                       'devices': sorted({row['runtime_stats'].get('device', 'unknown') for row in valid})})
    (ROOT / 'research/trace-summary.json').write_text(json.dumps(output, indent=2) + '\n')
    for item in output:
        print(f"L{item['level']}P{item['problem_id']} {item['model']}: correct={item['reported_correct']}, valid_timed={item['valid_timed_attempts']}, unchanged={item['consecutive_unchanged_code']}, best_turn={item['best_valid_turn']}, last_valid={item['last_turn_valid']}")


if __name__ == '__main__':
    main()
