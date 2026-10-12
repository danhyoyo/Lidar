# RC-SGFPN gate modes

`model.rc_gate_mode` selects range conditioning for `neck_type="rc_sgfpn"`.
The default is `"mul"`, preserving existing configs, training weights and
multiplicative deploy buffers. The same option is supported by the existing
bidirectional RC neck, without changing its topology.

Let `z = DWConv(L + U)` and `b(r) = Conv1x1(FourierRangeEmbedding(r))`:

- `mul`: `g = 2 * sigmoid(z * (1 + 0.5 * tanh(b(r))))`.
- `add`: `g = 2 * sigmoid(z + b(r))`.
- Both modes return `U + g * L` and start as `U + L` with zero-initialized gates.

The additive prior can change whether the lateral feature is suppressed or
amplified. This is an experimental option; adding it does not establish an AP
improvement over multiplicative gating.

## Notebook custom controls

Set these variables before the cell constructs `custom_overrides`:

```python
PRESET = "custom"
SCALE_GATED_FPN = True
NECK_FUSION_CHANNELS = 24
DETAIL_PATH = False

# A0: SG-FPN baseline
NECK_TYPE = "scale_gated_fpn"
RC_GATE_MODE = "mul"
CUSTOM_RUN_NAME = "neck-a0-sgfpn-s42"

# A1: replace the three lines above with:
# NECK_TYPE = "rc_sgfpn"
# RC_GATE_MODE = "mul"
# CUSTOM_RUN_NAME = "neck-a1-rcmul-s42"

# A2: replace the three lines above with:
# NECK_TYPE = "rc_sgfpn"
# RC_GATE_MODE = "add"
# CUSTOM_RUN_NAME = "neck-a2-rcadd-s42"
```

Keep the encoder, backbone stages, head, loss, augmentation, split, training
schedule and seed matched. Re-run the configuration cell and dependent cells
after editing controls. A new ablation needs a new run directory, because
notebooks automatically resume checkpoints found in the selected run.
Automatic run naming also distinguishes additive RC with a `rcadd` tag.

## Checkpoints and deployment

Explicit `mul` and an omitted mode have identical checkpoint identities.
`add` has a distinct identity, so mode changes are rejected by checkpoint
validation instead of silently reinterpreting training weights. A raw training
state dict still needs the correct mode in its accompanying config, since
both modes have identical training parameter names and shapes.

Deploy caches `static_spatial_scale` for `mul` and `static_spatial_bias` for
`add`. Equivalence is verified at the configured cached grid size. The existing
interpolation fallback for other grid sizes is retained; it is not an exact
replacement for regenerating Fourier embeddings at a new resolution.
