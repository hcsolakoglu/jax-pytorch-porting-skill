# Third-party provenance and licenses

Original contributions in this repository are MIT licensed. This does not relicense third-party material or imply endorsement by its authors.

| Material | Provenance | License |
|---|---|---|
| `validation/originals/flax_imagenet/` | Pinned `google/flax` ImageNet model/training source | [Apache-2.0](validation/originals/flax_imagenet/LICENSE) |
| `validation/ports/resnet_torch.py` | Native port derived from that Flax source; modifications are documented in its header and validation notes | Apache-2.0; retain original attribution and linked license |
| `validation/originals/torch_language_model/` | Pinned `pytorch/examples` source | [BSD-3-Clause](validation/originals/torch_language_model/LICENSE), copyright 2017 Pytorch contributors |
| `validation/ports/rnn_jax.py` | Native implementation derived from the documented source behavior and original model | BSD-3-Clause; retain original attribution and linked license |

Exact revisions, source URLs and SHA256 hashes appear in [originals/manifest.json](validation/originals/manifest.json). Original files remain unchanged. No pretrained model weights or datasets are redistributed.

Third-party skill snapshots and raw agent traces are ignored local research inputs, not part of the distributable package. Research summaries cite their sources and do not claim ownership of those projects. Source hashes and concise analysis are retained for reproducibility. Public availability does not remove licensing or privacy obligations.

The skill ZIP contains only our guidance, metadata, NumPy parity helper and MIT license. JAX, PyTorch, Flax, Optax, benchmark projects and harness names are used descriptively; their maintainers have not endorsed this skill.
