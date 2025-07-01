[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Cheetah Demos

<img src="https://github.com/desy-ml/cheetah-demos/raw/master/images/logo.png" align="right" width="25%"/>

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
@article{kaiser2024cheetah,
    title        = {Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations},
    author       = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = {May},
    journal      = {Phys. Rev. Accel. Beams},
    publisher    = {American Physical Society},
    volume       = 27,
    pages        = {054601},
    doi          = {10.1103/PhysRevAccelBeams.27.054601},
    url          = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.27.054601},
    issue        = 5,
    numpages     = 17
}
```
