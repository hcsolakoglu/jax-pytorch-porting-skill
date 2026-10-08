from tools.trace_snapshot import timing_eligible


def test_timing_requires_correct_finite_positive_measurement():
    assert timing_eligible({"correctness": True, "runtime": 0.01})
    for runtime in (-1, 0, float("inf"), float("nan"), None, True):
        assert not timing_eligible({"correctness": True, "runtime": runtime})
    assert not timing_eligible({"correctness": False, "runtime": 0.01})
    assert not timing_eligible(
        {"correctness": True, "runtime": 0.01, "metadata": {"performance_error": "timeout"}}
    )
