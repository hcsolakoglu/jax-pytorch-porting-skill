"""Original PyTorch language-model inference/training versus its native JAX port."""
from __future__ import annotations

import json
import tempfile
import warnings

from validation.environment import ROOT, original_module

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization

from validation.ports.rnn_jax import RNNModel, sgd_step
from validation.reporting import Report


def state_arrays(model):
    return {name: np.array(value.detach().numpy(), copy=True) for name, value in model.state_dict().items()}


def make_hidden(model, batch, random):
    value = random.normal(0, 0.1, (model.nlayers, batch, model.nhid)).astype(np.float32)
    if model.rnn_type == 'LSTM':
        other = random.normal(0, 0.1, value.shape).astype(np.float32)
        return (torch.from_numpy(value), torch.from_numpy(other)), (jnp.asarray(value), jnp.asarray(other))
    return torch.from_numpy(value), jnp.asarray(value)


def check_hidden(report, name, source_hidden, target_hidden):
    for index, (expected, actual) in enumerate(zip(jax.tree_util.tree_leaves(source_hidden), jax.tree_util.tree_leaves(target_hidden), strict=True)):
        report.check(f'{name}/{index}', expected.detach().numpy(), np.asarray(actual), 'rnn_forward')


def source_step(source, hidden, tokens, labels):
    source.train()
    hidden = jax.tree_util.tree_map(lambda value: value.detach(), hidden)
    source.zero_grad(set_to_none=True)
    probabilities, next_hidden = source(torch.from_numpy(tokens), hidden)
    loss = torch.nn.functional.nll_loss(probabilities, torch.from_numpy(labels).long())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(source.parameters(), 0.25)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(parameter.grad, alpha=-0.1)
    return next_hidden, loss.detach().numpy(), norm.detach().numpy()


def run_model(report, kind, tied=False):
    original = original_module('torch_language_model')
    random = np.random.default_rng(772)
    torch.manual_seed(382)
    width, hidden_width = (6, 6) if tied else (5, 7)
    source = original.RNNModel(kind, 17, width, hidden_width, 2, dropout=0, tie_weights=tied)
    target = RNNModel(kind, 17, width, hidden_width, 2, dropout=0, tie_weights=tied)
    params = target.from_state_dict(state_arrays(source))
    prefix = kind + ('-tied' if tied else '')
    source.eval()
    compiled = jax.jit(target.apply)
    for sequence, batch in [(5, 3), (1, 1)]:
        tokens = random.integers(0, 17, (sequence, batch), dtype=np.int64)
        target.validate_inputs(tokens)
        source_hidden, target_hidden = make_hidden(target, batch, random)
        with torch.no_grad():
            expected, source_next = source(torch.from_numpy(tokens), source_hidden)
        actual, target_next = compiled(params, jnp.asarray(tokens), target_hidden)
        report.check(f'{prefix}/forward/{sequence}/{batch}', expected.numpy(), np.asarray(actual), 'rnn_forward')
        check_hidden(report, f'{prefix}/hidden/{sequence}/{batch}', source_next, target_next)
    tokens = random.integers(0, 17, (5, 3), dtype=np.int64)
    labels = random.integers(0, 17, tokens.size, dtype=np.int64)
    source_hidden, target_hidden = make_hidden(target, 3, random)
    source_hidden = jax.tree_util.tree_map(lambda value: value.requires_grad_(True), source_hidden)
    source.zero_grad(set_to_none=True)
    expected, _ = source(torch.from_numpy(tokens), source_hidden)
    source_loss = torch.nn.functional.nll_loss(expected, torch.from_numpy(labels))
    source_loss.backward()

    def loss_fn(parameters, hidden):
        outputs, _ = target.apply(parameters, jnp.asarray(tokens), hidden)
        return -jnp.take_along_axis(outputs, jnp.asarray(labels)[:, None], axis=1).mean()

    loss, (gradients, hidden_gradients) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))(params, target_hidden)
    report.check(f'{prefix}/loss', source_loss.detach().numpy(), np.asarray(loss), 'loss')
    source_parameters = dict(source.named_parameters())
    assert set(source_parameters) == set(params), 'Unique parameter identities differ'
    for name, parameter in source_parameters.items():
        report.check(f'{prefix}/gradient/{name}', parameter.grad.numpy(), np.asarray(gradients[name]), 'rnn_gradient')
    for index, (expected, actual) in enumerate(zip(jax.tree_util.tree_leaves(source_hidden), jax.tree_util.tree_leaves(hidden_gradients), strict=True)):
        report.check(f'{prefix}/hidden-gradient/{index}', expected.grad.numpy(), np.asarray(actual), 'rnn_gradient')
    if tied:
        assert source.encoder.weight is source.decoder.weight
        assert 'decoder.weight' not in params
        report.notes.append('Tied LSTM uses one JAX parameter leaf; its gradient matches both source uses combined.')
    train = jax.jit(lambda p, h, x, y: sgd_step(target, p, h, x, y, learning_rate=0.1, max_norm=0.25))
    source_hidden, target_hidden = make_hidden(target, 3, random)
    for step, sequence in enumerate([5, 5, 2]):
        tokens = random.integers(0, 17, (sequence, 3), dtype=np.int64)
        labels = random.integers(0, 17, tokens.size, dtype=np.int64)
        source_hidden, expected_loss, expected_norm = source_step(source, source_hidden, tokens, labels)
        params, target_hidden, actual_loss, actual_norm = train(params, target_hidden, jnp.asarray(tokens), jnp.asarray(labels))
        report.check(f'{prefix}/train/{step}/loss', expected_loss, np.asarray(actual_loss), 'loss')
        report.check(f'{prefix}/train/{step}/gradient-norm', expected_norm, np.asarray(actual_norm), 'rnn_gradient')
        check_hidden(report, f'{prefix}/train/{step}/hidden', source_hidden, target_hidden)
        for name, parameter in source.named_parameters():
            report.check(f'{prefix}/train/{step}/parameter/{name}', parameter.detach().numpy(), np.asarray(params[name]), 'parameter_update')

    # A full small continuation state, not just a model-only save/load.
    checkpoint = {'params': params, 'hidden': target_hidden, 'step': np.asarray(3, np.int32),
                  'rng_data': jax.random.key_data(jax.random.key(6)), 'rng_dtype': 'threefry2x32'}
    template = {'params': target.init(jax.random.key(999)), 'hidden': target.init_hidden(3),
                'step': np.asarray(0, np.int32), 'rng_data': jax.random.key_data(jax.random.key(999)),
                'rng_dtype': 'threefry2x32'}
    (ROOT / '.work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / '.work', prefix='rnn-checkpoint-') as temporary:
        path = ROOT / temporary / 'state.msgpack'
        path.write_bytes(serialization.to_bytes(checkpoint))
        restored = serialization.from_bytes(template, path.read_bytes())
    for name, value in params.items():
        report.check(f'{prefix}/checkpoint/{name}', np.asarray(value), np.asarray(restored['params'][name]))
    report.check(f'{prefix}/checkpoint/step', checkpoint['step'], restored['step'])
    resumed_key = jax.random.wrap_key_data(restored['rng_data'], dtype=restored['rng_dtype'])
    report.check(f'{prefix}/checkpoint/next-random-draw', np.asarray(jax.random.normal(jax.random.key(6), (4,), dtype=jnp.float32)),
                 np.asarray(jax.random.normal(resumed_key, (4,), dtype=jnp.float32)))
    source_hidden, expected_loss, _ = source_step(source, source_hidden, tokens, labels)
    resumed, resumed_hidden, resumed_loss, _ = train(restored['params'], restored['hidden'], jnp.asarray(tokens), jnp.asarray(labels))
    report.check(f'{prefix}/resume/loss', expected_loss, np.asarray(resumed_loss), 'loss')
    check_hidden(report, f'{prefix}/resume/hidden', source_hidden, resumed_hidden)
    for name, parameter in source.named_parameters():
        report.check(f'{prefix}/resume/{name}', parameter.detach().numpy(), np.asarray(resumed[name]), 'parameter_update')


def dropout_probe(report):
    original = original_module('torch_language_model')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        source = original.RNNModel('LSTM', 11, 4, 4, 1, dropout=0.25)
    target = RNNModel('LSTM', 11, 4, 4, 1, dropout=0.25)
    params = target.from_state_dict(state_arrays(source))
    random = np.random.default_rng(829)
    tokens = random.integers(0, 11, (3, 2), dtype=np.int64)
    masks = tuple(random.random((3, 2, 4)) < 0.75 for _ in range(2))

    class ExplicitDropout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0

        def forward(self, values):
            mask = torch.from_numpy(masks[self.index])
            self.index += 1
            return values * mask.to(values.dtype) / 0.75

    source.drop = ExplicitDropout()
    source.train()
    with torch.no_grad():
        expected, _ = source(torch.from_numpy(tokens), source.init_hidden(2))
    actual, _ = target.apply(params, jnp.asarray(tokens), target.init_hidden(2), train=True,
                             masks=tuple(jnp.asarray(mask) for mask in masks))
    report.check('dropout/injected-mask-forward', expected.numpy(), np.asarray(actual), 'rnn_forward')
    native = jax.jit(lambda key: target.apply(params, jnp.asarray(tokens), target.init_hidden(2), train=True, key=key)[0])
    first = np.asarray(native(jax.random.key(23)))
    report.check('dropout/repeated-key', first, np.asarray(native(jax.random.key(23))))
    assert not np.array_equal(first, np.asarray(native(jax.random.key(24))))
    try:
        target.validate_inputs(np.array([[-1]], dtype=np.int64))
    except IndexError:
        pass
    else:
        raise AssertionError('Invalid token was silently accepted')
    report.notes.extend(['Explicit diagnostic dropout seam changes only source.drop, never original files or recurrent equations.',
                         'Single-layer input/output dropout masks match exactly; multi-layer native fused dropout distributions remain untested.',
                         'Native JAX RNG repeatability and distinct-key behavior checked separately from cross-framework masks.',
                         'Host token validation prevents silent JAX out-of-bounds gather behavior.'])


def main():
    torch.set_num_threads(1)
    report = Report('rnn-training')
    for kind in ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'):
        run_model(report, kind)
        print(f'{kind}: inference, gradients, short training and resume PASS', flush=True)
    run_model(report, 'LSTM', tied=True)
    dropout_probe(report)
    report.notes.extend(['Five configurations: four recurrent families plus tied-weight LSTM.',
                         'Three training chunks including a short final chunk, followed by one resumed step per configuration.',
                         'Original PyTorch model classes executed; training follows original NLL, clipping and manual SGD equations.',
                         'Pure native JAX scan-based execution; no PyTorch callback or target-port reference.',
                         'CPU FP32 only; no corpus perplexity, long convergence, GPU or TPU claim.'])
    report.save(status='PASS')
    print(json.dumps({'status': 'PASS', 'comparisons': len(report.records), 'report': 'validation/results/rnn-training.json'}))


if __name__ == '__main__':
    main()
