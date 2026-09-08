"""Bounded cross-architecture probes; official operators, no target-port references."""
from validation.environment import ROOT

import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
import torch
import torch.nn.functional as functional


def main():
    torch.set_num_threads(1)
    results = {'evidence_class': 'OBSERVED', 'backend': 'CPU', 'versions': {
        'torch': torch.__version__, 'jax': jax.__version__, 'optax': optax.__version__}, 'checks': {}}
    checks = results['checks']
    x = np.linspace(-4, 4, 257, dtype=np.float64)
    exact = functional.gelu(torch.from_numpy(x)).numpy()
    default = np.asarray(jax.nn.gelu(jnp.asarray(x)))
    aligned = np.asarray(jax.nn.gelu(jnp.asarray(x), approximate=False))
    checks['gelu_default_max_abs'] = float(np.max(np.abs(exact-default)))
    checks['gelu_aligned_max_abs'] = float(np.max(np.abs(exact-aligned)))
    assert checks['gelu_default_max_abs'] > 1e-4
    np.testing.assert_allclose(exact, aligned, atol=1e-14, rtol=1e-13)

    random = np.random.default_rng(941)
    q, k, v = [random.normal(size=(2, 3, 5, 4)).astype(np.float32) for _ in range(3)]
    mask = np.tril(np.ones((1, 1, 5, 5), dtype=bool))
    tensors = [torch.from_numpy(value) for value in (q, k, v)]
    torch_attention = functional.scaled_dot_product_attention(*tensors, attn_mask=torch.from_numpy(mask)).numpy()
    jax_attention = np.asarray(jax.nn.dot_product_attention(
        *[jnp.asarray(value.transpose(0, 2, 1, 3)) for value in (q, k, v)],
        mask=jnp.asarray(mask), implementation='xla')).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(torch_attention, jax_attention, atol=2e-6, rtol=2e-5)
    checks['attention_layout_and_causal_mask_max_abs'] = float(np.max(np.abs(torch_attention-jax_attention)))
    # Required negative control: inverting an allowed-position mask changes outputs.
    inverted = functional.scaled_dot_product_attention(*tensors, attn_mask=torch.from_numpy(~mask)).numpy()
    assert not np.allclose(torch_attention, inverted, atol=2e-6, rtol=2e-5)
    empty = np.zeros_like(mask)
    torch_empty = functional.scaled_dot_product_attention(*tensors, attn_mask=torch.from_numpy(empty)).numpy()
    jax_empty = np.asarray(jax.nn.dot_product_attention(
        *[jnp.asarray(value.transpose(0, 2, 1, 3)) for value in (q, k, v)],
        mask=jnp.asarray(empty), implementation='xla')).transpose(0, 2, 1, 3)
    checks['all_masked'] = {'torch_all_zero': bool(np.all(torch_empty == 0)),
                           'jax_all_zero': bool(np.all(jax_empty == 0)),
                           'max_abs_difference': float(np.max(np.abs(torch_empty-jax_empty)))}

    initial = np.array([0.3, -0.8, 2.0], dtype=np.float64)
    parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, foreach=False)
    transform = optax.adamw(0.01, b1=0.9, b2=0.999, eps=1e-8, eps_root=0, weight_decay=0.1)
    values = jnp.asarray(initial)
    state = transform.init(values)
    differences = []
    for gradient in ([1e-12, -1e-9, 0.2], [0.7, -0.1, 0.0], [0.0, 0.0, 0.0]):
        grad = np.array(gradient, dtype=np.float64)
        parameter.grad = torch.from_numpy(grad.copy())
        optimizer.step()
        updates, state = transform.update(jnp.asarray(grad), state, values)
        values = optax.apply_updates(values, updates)
        slots = optimizer.state[parameter]
        for left, right in ((parameter.detach().numpy(), np.asarray(values)),
                            (slots['exp_avg'].numpy(), np.asarray(state[0].mu)),
                            (slots['exp_avg_sq'].numpy(), np.asarray(state[0].nu))):
            np.testing.assert_allclose(left, right, atol=2e-14, rtol=2e-12)
        assert int(slots['step']) == int(state[0].count)
        differences.append(float(np.max(np.abs(parameter.detach().numpy()-np.asarray(values)))))
    checks['adamw_parameter_and_moment_parity'] = {'steps': 3, 'max_abs_each_step': differences,
                                                'includes_epsilon_dominated_and_zero_gradients': True}

    def transform_case(value):
        return jax.lax.cond(value > 0, lambda z: z*z, lambda z: -z, value)

    def scan_case(values):
        def body(carry, value):
            return carry + transform_case(value), None
        return jax.lax.scan(body, jnp.float32(0), values)[0]

    scan_ir = str(jax.make_jaxpr(scan_case)(jnp.array([1, -2], dtype=jnp.float32)))
    vector_ir = str(jax.make_jaxpr(jax.vmap(transform_case))(jnp.array([1, -2], dtype=jnp.float32)))
    assert 'cond[' in scan_ir and 'select_n' in vector_ir
    checks['transform_structure'] = {'scan_retains_cond': True, 'batched_vmap_uses_select': True,
                                     'scope': 'JAXPR evidence, not a hardware timing claim'}
    value = jnp.array([0.2, -0.3, 0.7], dtype=jnp.float64)
    tangent = jnp.array([1.0, 2.0, -1.0], dtype=jnp.float64)
    fn = lambda z: jnp.sin(z) * z
    _, jvp = jax.jvp(fn, (value,), (tangent,))
    _, pullback = jax.vjp(fn, value)
    cotangent = jnp.array([-0.4, 0.8, 0.6], dtype=jnp.float64)
    duality_error = float(jnp.abs(jnp.vdot(cotangent, jvp)-jnp.vdot(pullback(cotangent)[0], tangent)))
    assert duality_error < 1e-14
    checks['jvp_vjp_duality_error'] = duality_error
    results['status'] = 'PASS'
    (ROOT/'validation/results/semantic-probes.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
