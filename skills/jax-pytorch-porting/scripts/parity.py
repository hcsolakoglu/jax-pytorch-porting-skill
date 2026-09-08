"""Strict NumPy parity diagnostics for bounded tensors; no framework imports.

Shapes and dtypes must match. Integer values are compared exactly without a
floating-point cast. Nonfinite values fail unless their matching pattern is
explicitly authorized. Metrics complement, rather than replace, a frozen budget.
"""
from __future__ import annotations

import numpy as np


class ParityError(AssertionError):
    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


def compare(reference, candidate, *, atol: float, rtol: float,
            allow_nonfinite: bool = False, allow_empty: bool = False) -> dict:
    """Return diagnostics or raise ParityError. Time/memory are O(element count)."""
    if not np.isfinite(atol) or not np.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ParityError(f'Structure differs: {reference.shape}/{reference.dtype} versus {candidate.shape}/{candidate.dtype}')
    metrics = {'shape': list(reference.shape), 'dtype': str(reference.dtype), 'elements': int(reference.size)}
    if reference.size == 0:
        if not allow_empty:
            raise ParityError('Empty comparison requires an explicit contract', metrics)
        return {**metrics, 'mismatches': 0, 'empty': True}
    if reference.dtype.kind in 'biu':
        mismatch = reference != candidate
        metrics['mismatches'] = int(np.count_nonzero(mismatch))
        if metrics['mismatches']:
            raise ParityError('Exact integer/boolean comparison failed', metrics)
        return metrics
    if reference.dtype.kind not in 'fc':
        raise TypeError(f'Unsupported comparison dtype: {reference.dtype}')
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not finite.all():
        equal_special = (reference == candidate) | (np.isnan(reference) & np.isnan(candidate))
        if not allow_nonfinite or not np.all(finite | equal_special):
            raise ParityError('Unexpected or mismatched nonfinite values', metrics)
    dtype = np.complex128 if reference.dtype.kind == 'c' else np.float64
    expected = reference[finite].astype(dtype, copy=False)
    actual = candidate[finite].astype(dtype, copy=False)
    if expected.size == 0:
        return {**metrics, 'mismatches': 0, 'finite_elements': 0}
    error = np.abs(actual - expected)
    budget = atol + rtol * np.abs(expected)
    mismatch = error > budget
    norms = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    cosine = float(np.real(np.vdot(expected, actual)) / norms) if norms else float(np.array_equal(expected, actual))
    metrics.update(max_abs=float(error.max()), mean_abs=float(error.mean()),
                   p95_abs=float(np.percentile(error, 95)),
                   rms_abs=float(np.sqrt(np.mean(error * error))), cosine=cosine,
                   mismatches=int(np.count_nonzero(mismatch)), atol=float(atol), rtol=float(rtol))
    if metrics['mismatches']:
        raise ParityError(f'{metrics["mismatches"]} values exceed the numerical budget; max_abs={metrics["max_abs"]:.8g}', metrics)
    return metrics
