import copy
import json

import pytest

from tools.render_comparison import ROOT, validated_entries


def test_comparison_is_complete_and_stably_ranked():
    data = json.loads((ROOT/'research/comparison-input.json').read_text())
    rows = validated_entries(data)
    assert len(rows) == 25 and sum(map(len, (row['scores'] for row in rows))) == 500
    assert sum(rows[0]['scores']) >= sum(rows[-1]['scores'])
    broken = copy.deepcopy(data)
    broken['entries'][0]['scores'][0] = True
    with pytest.raises(ValueError):
        validated_entries(broken)
    broken = copy.deepcopy(data)
    broken['entries'][1] = broken['entries'][0]
    with pytest.raises(ValueError):
        validated_entries(broken)
