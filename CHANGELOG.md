<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Changelog

All notable changes to **ScireX** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `norm="group_norm_legacy"` and `"batch_norm_legacy"` for `FNOBlocks`, `FNO` and
  `GINO` (`fno_norm=...`), and `legacy_layout=True` for
  `scirex.layers.common.normalization.BatchNorm`: the 1.0.0 behaviour of
  `group_norm` / `batch_norm`, kept to run models built or trained with 1.0.0.

### Fixed

- `FNOBlocks` with `norm="group_norm"` or `"batch_norm"` (so also `FNO` and `GINO`
  through `norm` / `fno_norm`) and `scirex.layers.common.normalization.BatchNorm`
  now index their parameters by the channel axis of the channels-first
  `(batch, channels, *spatial)` input. Before, Flax treated the last spatial axis as
  the feature axis, so the variables had one entry per column of that axis instead
  of one per channel and only worked at the resolution they were created at. For
  `group_norm` (one group) this affected the scale and bias only; its statistics,
  over all channels and positions, were already right. For `batch_norm` the running
  mean and variance, and so the normalisation itself, were also per column.

### Changed

- **Behaviour change:** for these two norms the scale and bias (and, for
  `batch_norm`, the batch statistics) now have shape `(channels,)` instead of
  `(last spatial size,)`. A freshly initialised `group_norm` model gives the same
  outputs as before up to rounding (scale 1, bias 0); trained `group_norm` models
  and every `batch_norm` model give different ones.
- `FNO` and `GINO` are now class version 0.2.0. `BaseModel.from_checkpoint` calls the
  new class hook `_migrate_init_kwargs`, and for a 0.1.0 (1.0.0) checkpoint `FNO` and
  `GINO` map `norm` / `fno_norm` `"group_norm"` and `"batch_norm"` to the legacy
  options, so those checkpoints load and give the same outputs as before. The
  chip-cooling benchmark's `config.json` now has a `"format": 2`, and one without it
  is rebuilt with the legacy norm in the same way.
- Parameters in the 1.0.0 layout applied with the per-channel norm (for example a
  1.0.0 params file restored into a freshly built model) raise a `ValueError` that
  names the legacy option. If the last spatial size equals the channel count the
  shapes match, so no error is raised and the outputs are wrong; use the legacy
  option there too.
- The `GINODarcy2d` and `GINOPoisson2d` presets and the chip-cooling benchmark set
  `fno_norm="group_norm_legacy"`. The API policy does not let a minor release change
  the shapes of the parameters a released preset builds; it also keeps the
  published Darcy, Darcy physics-only and chip-cooling checkpoints loading.

## [1.0.0] - 2026-11-12

First release of ScireX: a JAX/Flax library of neural operators for scientific machine
learning, succeeding the original [SciREX](https://github.com/zenteiq-sciml/SciREX) research
framework (a separate project from the AiREX Lab at IISc). It has a defined public API: the
stable names, the three stability tiers, the versioning rules and the deprecation policy are
in [`doc/dev_guide/api_policy.md`](doc/dev_guide/api_policy.md), and the public names are
listed in the [API reference](doc/api/index.md).

### Highlights

- **Models:** FNO, WNO, GINO and Transolver in pure JAX/Flax, plus the experimental GIWO.
- **Physics-informed training:** the finite-element Ritz-energy loss (`Q4Galerkin`), which
  trains a model without labelled solutions. Equation-residual losses are experimental
  (`scirex.experimental`).
- **Benchmarks:** Darcy flow (data-driven and physics-only), pipe flow, 2D Navier-Stokes,
  lid-driven cavity, chip cooling, AirfRANS, ShapeNetCar and Car-CFD, at a matched parameter
  budget, with write-ups, dataset guides and the commands to reproduce each result.
- **Checkpoints:** the trained benchmark checkpoints are on Hugging Face, one repository per
  checkpoint (`zenteiq-ai/scirex-<model>-<problem>`). Lid-driven cavity has none because its
  data is not public.
- **Your own data:** `scripts/train.py` and `scripts/predict.py` train and run any of the four
  models on your own data in one command (see the *Use It* page).
- **Install and check:** pip (CPU, CUDA 12, TPU), Docker and source installs, with
  `python -m scirex.info` and `scripts/dev/check_platform.sh` to check a machine.
- **Documentation and tooling:** an MkDocs site with the API reference, theory guides and
  developer guides; hyperparameter tuning with Optuna and MLflow (`scirex.experimental.tuning`);
  Flax-native checkpointing and Weights & Biases logging; CI on Python 3.10, 3.11 and 3.12.

### Deprecated

These names still work and emit a `DeprecationWarning`. They are removed in 1.2.0 at the earliest.

| Name | Use instead |
|---|---|
| `scirex.training.append_grid_nhwc`, `append_grid_nwc`, `create_train_state`, `make_schedule`, `per_frame_relative_l2`, `plot_loss_curves`, `relative_l2_sum`, `resolve_dtype`, `write_metrics` | Not public API. They live in `scirex._internal.wno_train_utils` |
| `scirex.losses.ADDiff`, `AdvectionTerm`, `AdvectionTermAD`, `BatchedVariableCoefficientGalerkinResidualLossT3`, `BatchedVariableCoefficientGalerkinResidualLossT6`, `DarcyHybridLossGINO`, `DiffusionTermAD`, `DivergenceTerm`, `FourierDiff`, `GradientTerm`, `NavierStokesEqnLoss`, `PDEResidualLoss`, `ReactionTerm`, `ReactionTermAD`, `SourceTermAD`, `T3Galerkin`, `T6Galerkin`, `TransientTerm` | Experimental. Import them from `scirex.experimental.losses` |
| `scirex.training.SimpleDataLoader`, `Trainer`, `load_training_state`, `save_training_state` | Experimental. Import them from `scirex.experimental.training` |
| Module paths `scirex.data.web_utils`, `scirex.layers.common.einsum_utils`, `scirex.layers.common.segment_csr`, `scirex.training.torch_setup`, `scirex.training.trainer`, `scirex.training.training_state`, `scirex.training.wno_train_utils`, `scirex.distributed`, `scirex.geometry`, `scirex.training.patching`, `scirex.tuning`, `scirex.config.distributed`, `scirex.models.giwo` | The modules moved to `scirex._internal.*` (private) and `scirex.experimental.*`. The old path still imports the same module |

### Known limitations

- The tensor-GaLore projector (`scirex.training.optim`, not part of the stable API) cannot use
  `jax.checkpoint` or a warm restart with Tucker factors, because they call non-JAX TensorLy
  functions. Three tests mark this as an expected failure (`xfail`, six runs on CPU and GPU).
- `MeshDataModule` normalisation divides by `max - min`; a feature that is constant over the data
  gives a `RuntimeWarning` and NaN values.
- VTK 9.6 deprecates `GetData` on cell arrays. The AirfRANS loaders still use it and emit a
  `DeprecationWarning`. The `benchmark` extra allows `vtk>=9.0` with no upper bound, so a later
  VTK release may remove the method.
- Plain `optax` optimizers (e.g. `optax.adamw`) do not raise on complex-valued parameters; they
  run without error but update the imaginary parts in the wrong direction (see
  `scirex.training.optim.adamw_cx`, which handles this correctly). Nothing currently guards
  against building a plain-Optax optimizer over complex parameters by mistake.

[Unreleased]: https://github.com/zenteiq-research/scirex/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/zenteiq-research/scirex/releases/tag/v1.0.0
