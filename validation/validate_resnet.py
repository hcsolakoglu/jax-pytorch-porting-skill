"""Bounded validation of an original Flax ResNet18 port to native PyTorch."""
from __future__ import annotations

import argparse
import ast
from functools import partial
import json
import tempfile
from typing import Any

from validation.environment import ROOT, original_module

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import linen as nn, serialization
from flax.training import common_utils, train_state

from validation.ports.resnet_torch import ResNet, FlaxBatchNorm, SameConv, canonical_to_torch, flatten_tree, load_flax, torch_key
from validation.reporting import Report


class SourceTrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: Any = None


def source_training_functions() -> dict:
    # Execute only these unchanged original AST functions, not TensorFlow launchers.
    path = ROOT / 'validation/originals/flax_imagenet/train.py'
    tree = ast.parse(path.read_text())
    names = {'cross_entropy_loss', 'compute_metrics', 'train_step'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Original training function allowlist no longer matches')
    namespace = {'jax': jax, 'jnp': jnp, 'lax': jax.lax, 'optax': optax,
                 'common_utils': common_utils, 'NUM_CLASSES': 1000}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def compare_state(report: Report, prefix: str, state, target: ResNet):
    target_state = target.state_dict()
    for collection, values in {'params': state.params, 'batch_stats': state.batch_stats}.items():
        for path, value in flatten_tree(values).items():
            report.check(prefix + '/' + collection + '/' + '/'.join(path), canonical_to_torch(path, value),
                         target_state[torch_key(path)].numpy(), 'state' if collection == 'batch_stats' else 'parameter_update')


def torch_train_step(model, optimizer, images, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(torch.from_numpy(images))
    classification = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels).long())
    penalty = sum(parameter.square().sum() for parameter in model.parameters() if parameter.ndim > 1) * 0.00005
    (classification + penalty).backward()
    optimizer.step()
    return classification.detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    original = original_module('flax_imagenet', 'models.py')
    report = Report('resnet-training' if args.training else 'resnet-forward')
    random = np.random.default_rng(821)
    images = random.normal(size=(2, 17, 19, 3)).astype(np.float32)
    source = original.ResNet18(num_classes=1000, num_filters=4)
    variables = source.init(jax.random.key(27), jnp.asarray(images))
    target = ResNet(num_filters=4)
    mapping = load_flax(target, variables)
    (ROOT / 'validation/results/resnet-mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    target.eval()
    reference_eval = jax.jit(lambda variables, x: source.apply(variables, x, train=False))
    with torch.no_grad():
        report.check('original-initialization/forward', np.asarray(reference_eval(variables, images)), target(torch.from_numpy(images)).numpy(), 'resnet_forward')
    # A separate diagnostic fixture activates all originally zero residual branches.
    params = jax.tree_util.tree_map(lambda value: value, variables['params'])
    for name, block in params.items():
        if name.startswith('ResNetBlock_'):
            block['BatchNorm_1']['scale'] = jnp.full_like(block['BatchNorm_1']['scale'], 0.3)
    variables = {'params': params, 'batch_stats': variables['batch_stats']}
    load_flax(target, variables)
    captured = {}
    handles = []
    for name, module in target.named_modules():
        if isinstance(module, (SameConv, FlaxBatchNorm, torch.nn.Linear)) or name == 'conv_init':
            def hook(_module, _inputs, output, key=name):
                values = output.detach()
                captured[key] = values.permute(0, 2, 3, 1).numpy() if values.ndim == 4 else values.numpy()
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            output = target(torch.from_numpy(images)).numpy()
        expected, intermediates = source.apply(variables, images, train=False,
            capture_intermediates=lambda module, method: isinstance(module, (nn.Conv, nn.BatchNorm, nn.Dense)),
            mutable=['intermediates'])
        report.check('active-residual/forward', np.asarray(expected), output, 'resnet_forward')
        for path, values in flatten_tree(intermediates['intermediates']).items():
            key = '.'.join(path[:-1])
            report.check('activation/' + key, np.asarray(values[0]), captured[key], 'resnet_forward')
    finally:
        for handle in handles:
            handle.remove()
    for shape, scale in [((1, 18, 20, 3), 1.0), ((3, 9, 11, 3), 0.0), ((2, 21, 16, 3), 10.0)]:
        inputs = (random.normal(size=shape) * scale).astype(np.float32)
        with torch.no_grad():
            report.check(f'edge/{shape}/{scale}', np.asarray(reference_eval(variables, inputs)), target(torch.from_numpy(inputs)).numpy(), 'resnet_forward')
    # A real negative control must fail without changing frozen acceptance budgets.
    with torch.no_grad():
        target.Dense_0.bias[0].add_(0.01)
        changed = target(torch.from_numpy(images)).numpy()
    assert not np.allclose(np.asarray(reference_eval(variables, images)), changed, atol=3e-6, rtol=3e-5)
    load_flax(target, variables)
    report.notes.extend(['Native PyTorch execution; no source-framework fallback.',
                         'Original initialization and separately activated residual fixture both tested.',
                         'Source/target activation boundaries compared before gradients or training.',
                         'Injected decoder-bias negative control failed as intended.'])
    if not args.training:
        report.save(status='PASS')
        print(json.dumps({'status': 'PASS', 'comparisons': len(report.records), 'report': 'validation/results/resnet-forward.json'}))
        return

    # Source original train_step is actually executed through one CPU replica.
    learning_rate = lambda step: jnp.asarray(0.01, dtype=jnp.float32)
    source_state = SourceTrainState.create(apply_fn=source.apply, params=params,
        batch_stats=variables['batch_stats'], tx=optax.sgd(learning_rate, momentum=0.9, nesterov=True))
    train = jax.pmap(partial(source_training_functions()['train_step'], learning_rate_fn=learning_rate), axis_name='batch')
    optimizer = torch.optim.SGD(target.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    training_images = random.normal(size=(3, 33, 35, 3)).astype(np.float32)
    labels = np.asarray([1, 17, 983], dtype=np.int32)
    initial_source_state = source_state
    for step in range(2):
        replicated = jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), source_state)
        next_replicated, metrics = train(replicated, {'image': jnp.asarray(training_images[None]), 'label': jnp.asarray(labels[None])})
        source_state = jax.tree_util.tree_map(lambda value: value[0], next_replicated)
        loss = torch_train_step(target, optimizer, training_images, labels)
        report.check(f'train/{step}/cross_entropy', np.asarray(metrics['loss'][0]), loss, 'loss')
        compare_state(report, f'train/{step}', source_state, target)
        source_trace = flatten_tree(source_state.opt_state[0].trace)
        for path, trace in source_trace.items():
            parameter = dict(target.named_parameters())[torch_key(path)]
            report.check(f'train/{step}/momentum/' + '/'.join(path), canonical_to_torch(path, trace),
                         optimizer.state[parameter]['momentum_buffer'].numpy(), 'resnet_gradient')
            if step == 0:
                report.check('gradient/' + '/'.join(path), canonical_to_torch(path, trace), parameter.grad.numpy(), 'resnet_gradient')

    # Convert and restore training state into independently initialized target objects.
    restored_source = serialization.from_bytes(initial_source_state, serialization.to_bytes(source_state))
    fresh = ResNet(num_filters=4)
    load_flax(fresh, {'params': restored_source.params, 'batch_stats': restored_source.batch_stats})
    fresh_optimizer = torch.optim.SGD(fresh.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    for path, trace in flatten_tree(restored_source.opt_state[0].trace).items():
        parameter = dict(fresh.named_parameters())[torch_key(path)]
        fresh_optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(canonical_to_torch(path, trace))
    (ROOT / '.work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / '.work', prefix='resnet-checkpoint-') as temporary:
        checkpoint = ROOT / temporary / 'checkpoint.pt'
        torch.save({'model': fresh.state_dict(), 'optimizer': fresh_optimizer.state_dict(), 'step': int(restored_source.step)}, checkpoint)
        loaded = torch.load(checkpoint, weights_only=True)
        target = ResNet(num_filters=4)
        target.load_state_dict(loaded['model'], strict=True)
        optimizer = torch.optim.SGD(target.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        optimizer.load_state_dict(loaded['optimizer'])
        report.check('checkpoint/step', np.asarray(int(restored_source.step), np.int64), np.asarray(loaded['step'], np.int64))
        for key, value in fresh.state_dict().items():
            report.check('checkpoint/' + key, value.numpy(), target.state_dict()[key].numpy())
    replicated = jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), restored_source)
    next_state, metrics = train(replicated, {'image': jnp.asarray(training_images[None]), 'label': jnp.asarray(labels[None])})
    next_state = jax.tree_util.tree_map(lambda value: value[0], next_state)
    loss = torch_train_step(target, optimizer, training_images, labels)
    report.check('resume/cross_entropy', np.asarray(metrics['loss'][0]), loss, 'loss')
    compare_state(report, 'resume', next_state, target)
    target.eval()
    with torch.no_grad():
        expected = reference_eval({'params': next_state.params, 'batch_stats': next_state.batch_stats}, images)
        report.check('resume/evaluation', np.asarray(expected), target(torch.from_numpy(images)).numpy(), 'resnet_forward')
    report.notes.extend(['Two original source training steps plus one checkpoint-resumed step.',
                         'One physical CPU device only; not distributed or accelerator validation.',
                         'Compared CE, L2-regularized gradients, Nesterov momentum, parameters and running statistics.',
                         'Checkpoint test covers explicit model/SGD-state conversion and safe Torch reload, not an original Orbax checkpoint file.',
                         'No dataset accuracy, long convergence or production-scale performance claim.'])
    report.save(status='PASS')
    print(json.dumps({'status': 'PASS', 'comparisons': len(report.records), 'report': 'validation/results/resnet-training.json'}))


if __name__ == '__main__':
    main()
