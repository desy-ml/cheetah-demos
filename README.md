# Cheetah Demos

This repository contains a collection of demos accompanying the [_Cheetah_](https://github.com/desy-ml/cheetah) high-speed, differentiable beam dynamics simulation Python package.

For more information, see the paper where these demos were first introduced: [_Cheetah: Bridging the Gap Between Machine Learning and Particle Accelerator Physics with High-Speed, Differentiable Simulations_](https://arxiv.org/abs/2401.05815).

### Finding your way around

- `benchmark`: Various speed benchmarks for Cheetah and other simulation tools.
- `bo_prior`: Example of using a differentiable Cheetah model as a prior for Bayesian optimisation on a particle accelerator to improve tuning performance.
- `neural_network_space_charge_quad`: Implementation of a modular neural network surrogate model for high-speed computation of space charge effects through a quadrupole magnet.
- `reinforcement_learning`: Data and plotting code for example tuning performed by a neural network policy trained with reinforcement learning using a Cheetah simulation environment. The full RL example can be found in [_Learning-based Optimisation of Particle Accelerators Under Partial Observability Without Real-World Training_](https://proceedings.mlr.press/v162/kaiser22a.html).
- `system_identification`: Example of using Cheetah with gradient-based optimisation to identify the parameters of a particle accelerator model from noisy measurements.
- `tuning`: Example of using Cheetah with gradient-based optimisation to tune a particle accelerator subsection to a desired working point.

### Cite this repository

Please cite the original paper that these demos were introduced in:

```bibtex
@misc{kaiser2024cheetah,
  title         = {Cheetah: Bridging the Gap Between Machine Learning and Particle Accelerator Physics with High-Speed, Differentiable Simulations},
  author        = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and {Santamaria Garcia}, Andrea},
  year          = {2024},
  eprint        = {2401.05815},
  archiveprefix = {arXiv},
  primaryclass  = {physics.acc-ph}
}
```
