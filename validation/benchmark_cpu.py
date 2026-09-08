"""Small, paired CPU benchmarks after correctness gates; no accelerator claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time

from validation.environment import ROOT, original_module

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
os.environ.setdefault('MAX_JOBS', '1')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.work' / 'inductor-cache'))
# Match CPU capacity for both runtimes; only this process is constrained.
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])

import jax
import jax.numpy as jnp
import numpy as np
import torch

from validation.ports.resnet_torch import ResNet, load_flax
from validation.ports.rnn_jax import RNNModel
from validation.reporting import BUDGETS, parity
from validation.evidence import fingerprint, provenance


def summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {'median_ms': float(np.median(values)), 'iqr_ms': float(np.percentile(values, 75)-np.percentile(values, 25)),
            'p95_ms': float(np.percentile(values, 95)), 'samples_ms': samples}


def measure(function):
    start = time.perf_counter_ns()
    result = function()
    # Include completion of every output/carry, without a NumPy transfer in timed work.
    for value in jax.tree_util.tree_leaves(result):
        if hasattr(value, 'block_until_ready'):
            value.block_until_ready()
    return (time.perf_counter_ns()-start)/1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('resnet', 'rnn'), required=True)
    parser.add_argument('--compile-torch', action='store_true')
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    if not 10 <= args.samples <= 100:
        raise ValueError('Use 10 to 100 bounded samples')
    if args.compile_torch and args.model != 'resnet':
        raise ValueError('The compile experiment is scoped to the ResNet target')
    required = ROOT / 'validation/results' / ('resnet-training.json' if args.model == 'resnet' else 'rnn-training.json')
    prerequisite = json.loads(required.read_text())
    if (prerequisite['status'] != 'PASS' or
            prerequisite.get('provenance', {}).get('validation_sha256') != fingerprint()['validation_sha256']):
        raise RuntimeError('Correctness prerequisite is absent, failed or stale; rerun validation')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random = np.random.default_rng(294)
    if args.model == 'resnet':
        original = original_module('flax_imagenet', 'models.py')
        source = original.ResNet18(num_classes=1000, num_filters=4)
        inputs = random.normal(size=(3, 33, 35, 3)).astype(np.float32)
        j_inputs, t_inputs = jnp.asarray(inputs), torch.from_numpy(inputs)
        variables = source.init(jax.random.key(27), j_inputs)
        for name, block in variables['params'].items():
            if name.startswith('ResNetBlock_'):
                block['BatchNorm_1']['scale'] = jnp.full_like(block['BatchNorm_1']['scale'], 0.3)
        target = ResNet(num_filters=4).eval()
        load_flax(target, variables)
        source_function = jax.jit(lambda state, x: source.apply(state, x, train=False))
        source_call = lambda: source_function(variables, j_inputs)
        target_function = torch.compile(target, fullgraph=True) if args.compile_torch else target
        target_call = lambda: target_function(t_inputs)
        budget = 'resnet_forward'
        source_label, target_label = 'JAX jit', 'PyTorch Inductor' if args.compile_torch else 'PyTorch eager'
        configuration = {'shape': list(inputs.shape), 'width': 4, 'classes': 1000, 'active_residual_scale': 0.3}
    else:
        original = original_module('torch_language_model')
        torch.manual_seed(382)
        source = original.RNNModel('LSTM', 17, 6, 6, 2, dropout=0, tie_weights=True).eval()
        target = RNNModel('LSTM', 17, 6, 6, 2, dropout=0, tie_weights=True)
        params = target.from_state_dict({name: value.detach().numpy() for name, value in source.state_dict().items()})
        inputs = random.integers(0, 17, (17, 4), dtype=np.int64)
        target.validate_inputs(inputs)
        t_inputs, j_inputs = torch.from_numpy(inputs), jnp.asarray(inputs)
        hidden = tuple(random.normal(0, .1, (2, 4, 6)).astype(np.float32) for _ in range(2))
        t_hidden, j_hidden = tuple(map(torch.from_numpy, hidden)), tuple(map(jnp.asarray, hidden))
        source_call = lambda: source(t_inputs, t_hidden)
        target_function = jax.jit(target.apply)
        target_call = lambda: target_function(params, j_inputs, j_hidden)
        budget = 'rnn_forward'
        source_label, target_label = 'PyTorch eager native LSTM', 'JAX jit scan'
        configuration = {'sequence': 17, 'batch': 4, 'embedding': 6, 'hidden': 6, 'layers': 2, 'tied': True}
    with torch.inference_mode():
        first = {'source_first_call_ms': measure(source_call), 'target_first_call_ms': measure(target_call)}
        expected, actual = source_call(), target_call()
        expected_leaves = jax.tree_util.tree_leaves(expected)
        actual_leaves = jax.tree_util.tree_leaves(actual)
        checks = []
        for left, right in zip(expected_leaves, actual_leaves, strict=True):
            left = left.numpy() if isinstance(left, torch.Tensor) else np.asarray(left)
            right = right.numpy() if isinstance(right, torch.Tensor) else np.asarray(right)
            checks.append(parity.compare(left, right, **BUDGETS[budget]))
        for _ in range(5):
            measure(source_call)
            measure(target_call)
        samples = {'source': [], 'target': []}
        for index in range(args.samples):
            order = [('source', source_call), ('target', target_call)]
            if index % 2:
                order.reverse()
            for name, function in order:
                samples[name].append(measure(function))
        # Confirm repeated calls did not silently mutate fixed inference state.
        for left, right in zip(jax.tree_util.tree_leaves(source_call()),
                               jax.tree_util.tree_leaves(target_call()), strict=True):
            left = left.numpy() if isinstance(left, torch.Tensor) else np.asarray(left)
            right = right.numpy() if isinstance(right, torch.Tensor) else np.asarray(right)
            parity.compare(left, right, **BUDGETS[budget])
    paired_ratios = np.asarray(samples['source'])/np.asarray(samples['target'])
    # Descriptive paired bootstrap; one session, not independent deployment evidence.
    indices = random.integers(0, args.samples, (2000, args.samples))
    boot = np.median(paired_ratios[indices], axis=1)
    result = {'evidence_class': 'OBSERVED', 'scope': 'CPU FP32 synthetic tiny-model inference, one session',
              'source': source_label, 'target': target_label, 'configuration': configuration,
              'torch_version': torch.__version__, 'jax_version': jax.__version__, 'devices': str(jax.devices()),
              'cpu_threads': 1, 'affinity_cores': len(os.sched_getaffinity(0)), 'precision': 'FP32',
              'boundary': 'preplaced inputs and weights, model plus all output completion; no data loading or NumPy copy',
              'state': 'eval; immutable fixed input state reused on both sides',
              'warmup_calls_each': 5, 'alternating_order': True,
              'cold_cost_limit': 'First call includes execution and compilation/dispatch setup; imports, initialization and loading excluded.',
              **first, 'source_timing': summary(samples['source']), 'target_timing': summary(samples['target']),
              'paired_median_speedup': float(np.median(paired_ratios)),
              'paired_bootstrap_95pct': np.percentile(boot, [2.5, 97.5]).tolist(),
              'uncertainty_limit': 'Within-session bootstrap is descriptive; samples can be autocorrelated and host is shared.',
              'correctness': checks, 'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'memory_limit': 'Combined-process high-water RSS includes both runtimes and compilation; not per-model peak memory.',
              'no_regression_observed': bool(np.percentile(boot, 2.5) >= 1)}
    result['provenance'] = provenance()
    result['benchmark_sha256'] = hashlib.sha256((ROOT/'validation/benchmark_cpu.py').read_bytes()).hexdigest()
    result['baseline_scope'] = 'Matched bounded developer configurations, not an exhaustive source-tuning search'
    result['input_sha256'] = hashlib.sha256(inputs.tobytes()).hexdigest()
    suffix = '-inductor' if args.compile_torch else ''
    path = ROOT / 'validation/results' / f'benchmark-{args.model}{suffix}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ['source', 'target', 'paired_median_speedup', 'paired_bootstrap_95pct',
                     'source_first_call_ms', 'target_first_call_ms', 'process_peak_rss_mib', 'no_regression_observed']}, indent=2))


if __name__ == '__main__':
    main()
