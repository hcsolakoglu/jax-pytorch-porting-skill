"""Source-only calibration and documented semantic probes, before target ports."""
from __future__ import annotations

import json
from validation.environment import ROOT, original_module

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
from flax.traverse_util import flatten_dict


def main() -> None:
    torch.set_num_threads(1)
    original = original_module('flax_imagenet', 'models.py')
    model = original.ResNet18(num_classes=1000, num_filters=4)
    values = np.random.default_rng(114).normal(size=(2, 17, 19, 3)).astype('float32')
    inputs = jnp.asarray(values)
    variables = model.init(jax.random.key(27), inputs)
    forward = jax.jit(lambda state, x: model.apply(state, x, train=False))
    reference = np.asarray(forward(variables, inputs))
    repeated = np.asarray(forward(variables, inputs))
    model64 = original.ResNet18(num_classes=1000, num_filters=4, dtype=jnp.float64)
    output64 = np.asarray(jax.jit(lambda state, x: model64.apply(state, x, train=False))(
        variables, inputs.astype(jnp.float64)))
    active = jax.tree_util.tree_map(lambda value: value, variables)
    flat = flatten_dict(active['params'])
    zero_scales = []
    for path, value in flat.items():
        if path[-1] == 'scale' and not np.any(np.asarray(value)):
            zero_scales.append('/'.join(path))
    # Isolate the documented BatchNorm update discrepancy on identical data.
    x = jnp.asarray(np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / 10)
    norm = nn.BatchNorm(momentum=0.9, epsilon=1e-5)
    initial = norm.init(jax.random.key(0), x, use_running_average=False)
    j_output, new_state = norm.apply(initial, x, use_running_average=False, mutable=['batch_stats'])
    t_norm = torch.nn.BatchNorm2d(3, momentum=0.1, eps=1e-5)
    t_output = t_norm(torch.tensor(np.asarray(x)).permute(0, 3, 1, 2))
    difference = np.abs(reference.astype(np.float64) - output64)
    result = {
        'evidence_class': 'OBSERVED',
        'stage': 'source-only, before target implementation',
        'versions': {'jax': jax.__version__, 'torch': torch.__version__, 'numpy': np.__version__},
        'backend': str(jax.devices()), 'shape': list(inputs.shape), 'num_filters': 4,
        'source_repeat_exact': bool(np.array_equal(reference, repeated)),
        'source_fp32_vs_fp64_max_abs': float(difference.max()),
        'source_fp32_vs_fp64_mean_abs': float(difference.mean()),
        'zero_residual_scales': zero_scales,
        'parameter_shapes': {'/'.join(key): list(value.shape) for key, value in flat.items()},
        'batchnorm_forward_max_abs': float(np.max(np.abs(np.asarray(j_output) - t_output.detach().permute(0, 2, 3, 1).numpy()))),
        'flax_running_variance': np.asarray(new_state['batch_stats']['var']).tolist(),
        'torch_running_variance': t_norm.running_var.detach().tolist(),
        'notes': ['Torch momentum complemented to 0.1; native running variance still differs.',
                  'Zero residual scales are original source initialization, not a port defect.'],
    }
    destination = ROOT / 'validation' / 'results'
    destination.mkdir(exist_ok=True)
    (destination / 'source-calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'parameter_shapes'}, indent=2))


if __name__ == '__main__':
    main()
