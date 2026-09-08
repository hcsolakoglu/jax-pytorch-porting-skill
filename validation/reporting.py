"""Small evidence recorder shared by the two bounded validation projects."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

from validation.environment import ROOT

spec = importlib.util.spec_from_file_location('port_parity', ROOT / 'skills/jax-pytorch-porting/scripts/parity.py')
if spec is None or spec.loader is None:
    raise ImportError('Parity helper unavailable')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)
BUDGETS = json.loads((ROOT / 'validation/budgets.json').read_text())


class Report:
    def __init__(self, name: str):
        self.name = name
        self.started = time.monotonic()
        self.records = []
        self.notes = []
        self.status = 'RUNNING'

    def check(self, name: str, reference, candidate, budget: str = 'checkpoint') -> dict:
        try:
            result = parity.compare(reference, candidate, **BUDGETS[budget])
        except parity.ParityError as error:
            self.records.append({'name': name, 'status': 'FAIL', 'budget': budget, **error.metrics, 'error': str(error)})
            self.status = 'FAIL'
            self.save()
            raise
        self.records.append({'name': name, 'status': 'PASS', 'budget': budget, **result})
        return result

    def save(self, *, status: str | None = None) -> Path:
        if status is not None:
            self.status = status
        destination = ROOT / 'validation/results' / f'{self.name}.json'
        destination.parent.mkdir(exist_ok=True)
        payload = {'evidence_class': 'OBSERVED', 'status': self.status, 'budget_version': BUDGETS['version'],
                   'elapsed_seconds': time.monotonic()-self.started,
                   'comparison_records': len(self.records), 'notes': self.notes, 'checks': self.records}
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        return destination
