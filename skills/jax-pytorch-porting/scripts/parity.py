"""Strict NumPy parity diagnostics for bounded tensors; no framework imports.

Shapes/dtypes match exactly. Integers never pass through floating-point casts.
Nonfinite complex values require matching real and imaginary components, with
finite components exact in those exceptional slots. Extended floating precision
and diagnostic overflow fail closed rather than silently erase differences.
"""
from __future__ import annotations

import numpy as np


class ParityError(AssertionError):
    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


def _matching_special(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Compare exceptional complex slots componentwise, not by a single isnan bit."""
    if reference.dtype.kind == 'c':
        return (_matching_special(reference.real, candidate.real)
                & _matching_special(reference.imag, candidate.imag))
    return (reference == candidate) | (np.isnan(reference) & np.isnan(candidate))


def compare(reference, candidate, *, atol: float, rtol: float,
            allow_nonfinite: bool = False, allow_empty: bool = False) -> dict:
    """Return diagnostics or raise ParityError. Time/memory are O(element count).

    Use only bounded diagnostic tensors; stream large-model leaves separately.
    Float16/32/64 and complex64/128 are supported. Metrics do not calibrate a budget.
    """
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
    if reference.dtype.kind not in 'fc' or np.finfo(reference.dtype).nmant > 52:
        raise TypeError(f'Unsupported comparison dtype: {reference.dtype}')
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not finite.all():
        if not allow_nonfinite or not np.all(finite | _matching_special(reference, candidate)):
            raise ParityError('Unexpected or mismatched nonfinite components', metrics)
    dtype = np.complex128 if reference.dtype.kind == 'c' else np.float64
    expected = reference[finite].astype(dtype, copy=False)
    actual = candidate[finite].astype(dtype, copy=False)
    if expected.size == 0:
        return {**metrics, 'mismatches': 0, 'finite_elements': 0}
    # A finite-input overflow must never turn inf > inf into a false pass.
    with np.errstate(over='ignore', invalid='ignore'):
        error = np.abs(actual - expected)
        magnitude = np.abs(expected)
        budget = atol + rtol * magnitude
    if not np.all(np.isfinite(error) & np.isfinite(budget) & np.isfinite(magnitude)):
        raise ParityError('Comparison arithmetic overflow; use an independently justified scaled diagnostic', metrics)
    mismatch = error > budget
    maximum = float(error.max())
    scaled_error = error / maximum if maximum else error
    expected_scale, actual_scale = float(magnitude.max()), float(np.abs(actual).max())
    if expected_scale and actual_scale:
        left, right = expected / expected_scale, actual / actual_scale
        cosine = float(np.real(np.vdot(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right)))
        cosine = float(np.clip(cosine, -1, 1))
    else:
        cosine = float(np.array_equal(expected, actual))
    worst_flat = int(np.flatnonzero(finite)[int(error.argmax())])
    metrics.update(max_abs=maximum, mean_abs=float(maximum * scaled_error.mean()),
                   p95_abs=float(np.percentile(error, 95)), p99_abs=float(np.percentile(error, 99)),
                   rms_abs=float(maximum * np.sqrt(np.mean(scaled_error * scaled_error))),
                   cosine=cosine, worst_index=list(np.unravel_index(worst_flat, reference.shape)),
                   mismatches=int(np.count_nonzero(mismatch)), atol=float(atol), rtol=float(rtol))
    metrics['worst_index'] = [int(index) for index in metrics['worst_index']]
    if metrics['mismatches']:
        raise ParityError(f'{metrics["mismatches"]} values exceed numerical budget; max_abs={maximum:.8g}', metrics)
    return metrics
