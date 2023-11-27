from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn


# Test Problem
def simple_fodo_problem(
    input_param: Dict[str, float], incoming_beam: Optional[cheetah.Beam] = None
) -> Dict[str, float]:
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_xp=torch.tensor(1e-4),
            sigma_yp=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )

    fodo_segment = cheetah.Segment(
        [
            cheetah.Quadrupole(
                length=0.1, k1=torch.tensor(input_param["q1"]), name="Q1"
            ),
            cheetah.Drift(length=0.5, name="D1"),
            cheetah.Quadrupole(
                length=0.1, k1=torch.tensor(input_param["q2"]), name="Q2"
            ),
            cheetah.Drift(length=0.5, name="D1"),
        ]
    )

    out_beam = fodo_segment(incoming_beam)

    beam_size_mse = torch.mean(out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = torch.mean(out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    return {
        "mse": beam_size_mse.detach().numpy(),
        "log_mse": beam_size_mse.log().detach().numpy(),
        "mae": beam_size_mae.detach().numpy(),
        "log_mae": beam_size_mae.log().detach().numpy(),
    }


# Prior Mean Functions for BO
class FodoPriorMean(nn.Module):
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
                cheetah.Quadrupole(length=0.1, k1=torch.tensor(0.0), name="Q1"),
                cheetah.Drift(length=0.5, name="D1"),
                cheetah.Quadrupole(length=0.1, k1=torch.tensor(0.0), name="Q2"),
                cheetah.Drift(length=0.5, name="D1"),
            ]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        input_shape = X.shape
        X = X.reshape(-1, 2)
        y_s = torch.zeros(X.shape[:-1])
        for i, input_values in enumerate(X):
            self.segment.Q1.k1 = input_values[0].float()
            self.segment.Q2.k1 = input_values[1].float()
            out_beam = self.segment(self.incoming_beam)
            beam_size_mae = torch.mean(out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
            y_s[i] = beam_size_mae
        return y_s.reshape(input_shape[:-1])
