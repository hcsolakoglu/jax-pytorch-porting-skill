from pathlib import Path
import importlib.util

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('parity', Path(__file__).parents[1] / 'skills/jax-pytorch-porting/scripts/parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_exact_uint64_does_not_lose_low_bits():
    with pytest.raises(parity.ParityError):
        parity.compare(np.array([2**63], np.uint64), np.array([2**63+1], np.uint64), atol=100, rtol=1)


@pytest.mark.parametrize('candidate', [np.ones((2, 1)), np.ones(2, dtype=np.float32)])
def test_structure_precedes_broadcast(candidate):
    with pytest.raises(parity.ParityError):
        parity.compare(np.ones(2), candidate, atol=0, rtol=0)


def test_high_cosine_does_not_hide_scale_error():
    with pytest.raises(parity.ParityError) as error:
        parity.compare(np.array([1., 2.]), np.array([2., 4.]), atol=1e-8, rtol=1e-8)
    assert error.value.metrics['cosine'] > 0.999


def test_nonfinite_requires_matching_explicit_contract():
    values = np.array([np.nan, np.inf, -np.inf, 0.])
    with pytest.raises(parity.ParityError):
        parity.compare(values, values, atol=0, rtol=0)
    assert parity.compare(values, values, atol=0, rtol=0, allow_nonfinite=True)['mismatches'] == 0
    with pytest.raises(parity.ParityError):
        parity.compare(values, -values, atol=0, rtol=0, allow_nonfinite=True)


def test_empty_and_invalid_budgets_fail_closed():
    with pytest.raises(parity.ParityError):
        parity.compare(np.array([]), np.array([]), atol=0, rtol=0)
    with pytest.raises(ValueError):
        parity.compare(np.array([0.]), np.array([0.]), atol=np.nan, rtol=0)


def test_zero_and_complex_metrics_are_well_defined():
    assert parity.compare(np.zeros(2), np.zeros(2), atol=0, rtol=0)['cosine'] == 1
    value = np.array([1+2j, 3-4j])
    assert parity.compare(value, value, atol=0, rtol=0)['mismatches'] == 0


def test_complex_nan_does_not_hide_different_finite_component():
    with pytest.raises(parity.ParityError):
        parity.compare(np.array([complex(np.nan, 1)]), np.array([complex(np.nan, 2)]),
                       atol=0, rtol=0, allow_nonfinite=True)


def test_overflow_cannot_turn_mismatch_into_pass():
    with pytest.raises(parity.ParityError):
        parity.compare(np.array([1e308]), np.array([-1.7e308]), atol=0, rtol=2)


def test_large_equal_values_have_finite_diagnostics():
    import json
    result = parity.compare(np.array([1e308, 1e308]), np.array([1e308, 1e308]), atol=0, rtol=0)
    assert result['cosine'] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_extended_precision_is_not_silently_downcast():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip('This platform has no extended long-double precision')
    values = np.array([1], dtype=np.longdouble)
    different = np.nextafter(values, np.longdouble(2))
    with pytest.raises((TypeError, parity.ParityError)):
        parity.compare(values, different, atol=0, rtol=0)
