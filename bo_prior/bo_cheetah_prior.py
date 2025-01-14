from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Positive
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


# Test Problem
def simple_fodo_problem(
    input_param: Dict[str, float],
    incoming_beam: Optional[cheetah.Beam] = None,
    lattice_distances: Optional[Dict[str, float]] = {},
) -> Dict[str, float]:
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
    quad_length = torch.tensor(lattice_distances.get("quad_length", 0.1))
    drift_length = torch.tensor(lattice_distances.get("drift_length", 0.5))

    fodo_segment = cheetah.Segment(
        [
            cheetah.Quadrupole(
                length=quad_length, k1=torch.tensor(input_param["q1"]), name="Q1"
            ),
            cheetah.Drift(length=drift_length, name="D1"),
            cheetah.Quadrupole(
                length=quad_length, k1=torch.tensor(input_param["q2"]), name="Q2"
            ),
            cheetah.Drift(length=drift_length, name="D1"),
        ]
    )

    out_beam = fodo_segment(incoming_beam)

    beam_size_mse = 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    return {
        "mse": beam_size_mse.detach().numpy(),
        "log_mse": beam_size_mse.log().detach().numpy(),
        "mae": beam_size_mae.detach().numpy(),
        "log_mae": beam_size_mae.log().detach().numpy(),
    }


# Prior Mean Functions for BO
class FodoPriorMean(Mean):
    """FODO Lattice as a prior mean function for BO."""

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
        super().__init__()
        if incoming_beam is None:
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                sigma_x=torch.tensor(1e-4),
                sigma_y=torch.tensor(2e-3),
                sigma_xp=torch.tensor(1e-4),
                sigma_yp=torch.tensor(1e-4),
                energy=torch.tensor(100e6),
            )
        self.incoming_beam = incoming_beam
        self.segment = cheetah.Segment(
            [
                cheetah.Quadrupole(
                    length=torch.tensor(0.1), k1=torch.tensor(0.0), name="Q1"
                ),
                cheetah.Drift(length=torch.tensor(0.5), name="D1"),
                cheetah.Quadrupole(
                    length=torch.tensor(0.1), k1=torch.tensor(0.0), name="Q2"
                ),
                cheetah.Drift(length=torch.tensor(0.5), name="D2"),
            ]
        )

        # Introduce a fittable parameter for the lattice
        drift_length_constraint = Positive()
        self.register_parameter("raw_drift_length", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "drift_length_prior",
            SmoothedBoxPrior(0.2, 1.0),
            self._drift_length_param,
            self._set_drift_length,
        )
        self.register_constraint("raw_drift_length", drift_length_constraint)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.segment.D1.length = self.drift_length.float()
        self.segment.D2.length = self.drift_length.float()

        input_shape = X.shape
        X = X.reshape(-1, 2)
        y_s = torch.zeros(X.shape[:-1])
        for i, input_values in enumerate(X):
            self.segment.Q1.k1 = input_values[0].float()
            self.segment.Q2.k1 = input_values[1].float()
            out_beam = self.segment(self.incoming_beam)
            beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
            y_s[i] = beam_size_mae
        return y_s.reshape(input_shape[:-1])

    @property
    def drift_length(self):
        return self._drift_length_param(self)

    @drift_length.setter
    def drift_length(self, value: torch.Tensor):
        self._set_drift_length(self, value)

    # Strange hack to conform with gpytorch definitions
    def _drift_length_param(self, m):
        return m.raw_drift_length_constraint.transform(self.raw_drift_length)

    def _set_drift_length(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_drift_length)
        m.initialize(
            raw_drift_length=m.raw_drift_length_constraint.inverse_transform(value)
        )
