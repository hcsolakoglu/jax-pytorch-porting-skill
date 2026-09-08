"""Native JAX port of pytorch/examples word_language_model.RNNModel.

Original source is BSD-3-Clause licensed; attribution and license are retained in
validation/originals/torch_language_model/LICENSE. This implementation uses no
PyTorch runtime. It supports LSTM, GRU, tanh/ReLU RNN, stacked layers, tied weights,
explicit hidden state, dropout, differentiation and JIT on valid token inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RNNModel:
    rnn_type: str
    ntoken: int
    ninp: int
    nhid: int
    nlayers: int
    dropout: float = 0.5
    tie_weights: bool = False

    def __post_init__(self):
        if self.rnn_type not in ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'):
            raise ValueError('Unsupported recurrent model')
        if min(self.ntoken, self.ninp, self.nhid, self.nlayers) < 1 or not 0 <= self.dropout <= 1:
            raise ValueError('Invalid model dimensions or dropout probability')
        if self.tie_weights and self.ninp != self.nhid:
            raise ValueError('Tied weights require matching embedding and hidden dimensions')

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {'encoder.weight': (self.ntoken, self.ninp), 'decoder.bias': (self.ntoken,)}
        if not self.tie_weights:
            shapes['decoder.weight'] = (self.ntoken, self.nhid)
        gates = {'LSTM': 4, 'GRU': 3}.get(self.rnn_type, 1)
        for layer in range(self.nlayers):
            prefix = 'rnn.'
            shapes[f'{prefix}weight_ih_l{layer}'] = (gates * self.nhid, self.ninp if layer == 0 else self.nhid)
            shapes[f'{prefix}weight_hh_l{layer}'] = (gates * self.nhid, self.nhid)
            shapes[f'{prefix}bias_ih_l{layer}'] = (gates * self.nhid,)
            shapes[f'{prefix}bias_hh_l{layer}'] = (gates * self.nhid,)
        return shapes

    def init(self, key: jax.Array) -> dict[str, jax.Array]:
        shapes = self.parameter_shapes()
        keys = jax.random.split(key, len(shapes))
        params = {}
        for (name, shape), subkey in zip(shapes.items(), keys, strict=True):
            bound = 1 / math.sqrt(self.nhid) if name.startswith('rnn.') else 0.1
            params[name] = jnp.zeros(shape, jnp.float32) if name == 'decoder.bias' else jax.random.uniform(
                subkey, shape, dtype=jnp.float32, minval=-bound, maxval=bound)
        return params

    def from_state_dict(self, state: Mapping[str, object]) -> dict[str, jax.Array]:
        supplied = dict(state)
        if self.tie_weights:
            if 'decoder.weight' not in supplied or not np.array_equal(supplied['decoder.weight'], supplied['encoder.weight']):
                raise ValueError('Checkpoint does not contain equal tied embedding/decoder values')
            del supplied['decoder.weight']
        expected = self.parameter_shapes()
        if set(supplied) != set(expected):
            raise ValueError(f'Parameter keys differ: missing={set(expected)-set(supplied)}, extra={set(supplied)-set(expected)}')
        params = {}
        for name, shape in expected.items():
            array = np.asarray(supplied[name])
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f'Expected FP32 {name} with shape {shape}')
            params[name] = jnp.asarray(array)
        return params

    def init_hidden(self, batch: int, dtype=jnp.float32):
        if batch < 1:
            raise ValueError('Batch size must be positive')
        hidden = jnp.zeros((self.nlayers, batch, self.nhid), dtype)
        return (hidden, jnp.zeros_like(hidden)) if self.rnn_type == 'LSTM' else hidden

    def validate_inputs(self, tokens: np.ndarray) -> None:
        """Validate host inputs once before a compiled run; JAX gathers do not raise on all invalid indices."""
        values = np.asarray(tokens)
        if values.ndim != 2 or min(values.shape) < 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Expected a nonempty sequence-major integer token matrix')
        if values.min() < 0 or values.max() >= self.ntoken:
            raise IndexError('Token outside vocabulary')

    def apply(self, params: Mapping[str, jax.Array], tokens: jax.Array, hidden,
              *, train: bool = False, key: jax.Array | None = None, masks: tuple | None = None):
        """Apply to previously validated tokens. Dropout masks are an explicit diagnostic seam."""
        if train and self.dropout and key is None and masks is None:
            raise ValueError('Training dropout requires a key or explicit diagnostic masks')
        keys = jax.random.split(key, self.nlayers + 1) if key is not None else (None,) * (self.nlayers + 1)

        def drop(values, index):
            if not train or self.dropout == 0:
                return values
            if self.dropout == 1:
                return jnp.zeros_like(values)
            mask = masks[index] if masks is not None else jax.random.bernoulli(keys[index], 1-self.dropout, values.shape)
            if mask.shape != values.shape:
                raise ValueError('Diagnostic dropout mask shape differs from activation')
            return values * mask.astype(values.dtype) / (1-self.dropout)

        values = drop(params['encoder.weight'][tokens], 0)
        final_h, final_c = [], []
        for layer in range(self.nlayers):
            projected = values @ params[f'rnn.weight_ih_l{layer}'].T + params[f'rnn.bias_ih_l{layer}']
            recurrent_weight = params[f'rnn.weight_hh_l{layer}']
            recurrent_bias = params[f'rnn.bias_hh_l{layer}']

            def cell(carry, projected_input):
                previous = carry[0] if self.rnn_type == 'LSTM' else carry
                recurrent = previous @ recurrent_weight.T + recurrent_bias
                if self.rnn_type == 'LSTM':
                    input_gate, forget_gate, candidate, output_gate = jnp.split(projected_input + recurrent, 4, axis=-1)
                    state = jax.nn.sigmoid(forget_gate) * carry[1] + jax.nn.sigmoid(input_gate) * jnp.tanh(candidate)
                    output = jax.nn.sigmoid(output_gate) * jnp.tanh(state)
                    return (output, state), output
                if self.rnn_type == 'GRU':
                    ir, iz, inn = jnp.split(projected_input, 3, axis=-1)
                    hr, hz, hn = jnp.split(recurrent, 3, axis=-1)
                    reset = jax.nn.sigmoid(ir + hr)
                    update = jax.nn.sigmoid(iz + hz)
                    candidate = jnp.tanh(inn + reset * hn)
                    output = (1-update) * candidate + update * previous
                else:
                    activation = jnp.tanh if self.rnn_type == 'RNN_TANH' else jax.nn.relu
                    output = activation(projected_input + recurrent)
                return output, output

            initial = (hidden[0][layer], hidden[1][layer]) if self.rnn_type == 'LSTM' else hidden[layer]
            final, values = jax.lax.scan(cell, initial, projected)
            if self.rnn_type == 'LSTM':
                final_h.append(final[0])
                final_c.append(final[1])
            else:
                final_h.append(final)
            if layer < self.nlayers - 1:
                values = drop(values, layer + 1)
        values = drop(values, self.nlayers)
        decoder = params['encoder.weight'] if self.tie_weights else params['decoder.weight']
        logits = values @ decoder.T + params['decoder.bias']
        result = jax.nn.log_softmax(logits.reshape(-1, self.ntoken), axis=1)
        next_hidden = (jnp.stack(final_h), jnp.stack(final_c)) if self.rnn_type == 'LSTM' else jnp.stack(final_h)
        return result, next_hidden


def sgd_step(model: RNNModel, params, hidden, tokens, labels, *, learning_rate: float,
             max_norm: float, key=None):
    """Original example's truncated-backpropagation, clipped manual-SGD update."""
    hidden = jax.tree_util.tree_map(jax.lax.stop_gradient, hidden)

    def loss_fn(parameters):
        log_probabilities, next_hidden = model.apply(parameters, tokens, hidden, train=True, key=key)
        loss = -jnp.take_along_axis(log_probabilities, labels.reshape(-1, 1), axis=1).mean()
        return loss, next_hidden

    (loss, next_hidden), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
    norm = jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(gradients)))
    # Torch clip_grad_norm_ uses this epsilon; Optax's default clipping rule differs.
    coefficient = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    updated = jax.tree_util.tree_map(lambda parameter, gradient: parameter-learning_rate*coefficient*gradient, params, gradients)
    return updated, next_hidden, loss, norm
