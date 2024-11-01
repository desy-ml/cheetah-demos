from typing import Literal, Optional

import torch
from lightning import LightningModule
from torch import nn, optim


def target_weighted_mse_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute MSE loss, but weight each sample by the of the target value.
    """
    weights = torch.abs(target).mean(dim=1).unsqueeze(dim=1).repeat(1, 6)
    return torch.mean(weights * (predicted - target) ** 2)


class SpaceChargeQuadrupoleMLP(nn.Module):
    """
    MLP model for predicting the transfer of a `cheetah.ParameterBeam` under the
    consideration of space charge.
    """

    def __init__(
        self,
        num_hidden_layers: int = 3,
        hidden_layer_width: int = 100,
        hidden_activation: Optional[
            Literal["ReLU", "LeakyReLU", "Softplus", "Sigmoid", "Tanh"]
        ] = "ReLU",
        hidden_activation_args: dict = {},
        batch_normalization: bool = True,
    ):
        super().__init__()

        relevant_beam_parameter_dims = 8
        controls_dims = 2
        relevant_beam_parameter_delta_dims = 6

        self.input_layer = self.hidden_block(
            relevant_beam_parameter_dims + controls_dims,
            hidden_layer_width,
            activation=hidden_activation,
            activation_args=hidden_activation_args,
        )

        blocks = [
            self.hidden_block(
                in_features=hidden_layer_width,
                out_features=hidden_layer_width,
                activation=hidden_activation,
                activation_args=hidden_activation_args,
                batch_normalization=batch_normalization,
                bias=not batch_normalization,
            )
            for _ in range(num_hidden_layers - 1)
        ]
        self.hidden_net = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(
            hidden_layer_width, relevant_beam_parameter_delta_dims
        )

    def hidden_block(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[
            Literal["ReLU", "LeakyReLU", "Softplus", "Sigmoid", "Tanh"]
        ] = None,
        activation_args: dict = {},
        batch_normalization: bool = False,
    ):
        """
        Create a block of a linear layer and an activation, meant to be used as a hidden
        layer in this architecture.
        """
        if activation is None:
            activation_module = nn.Identity()
        else:
            activation_module = getattr(nn, activation)(**activation_args)

        return nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.BatchNorm1d(out_features) if batch_normalization else nn.Identity(),
            activation_module,
        )

    def forward(self, incoming, controls):
        x = torch.concatenate([incoming, controls], dim=1)
        x = self.input_layer(x)
        x = self.hidden_net(x)
        outgoing_parameters = self.output_layer(x)
        return outgoing_parameters


class SupervisedSpaceChargeQuadrupoleInference(LightningModule):
    """Model with supervised training for infering current profile at EuXFEL."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        num_hidden_layers: int = 3,
        hidden_layer_width: int = 100,
        hidden_activation: str = "relu",
        hidden_activation_args: dict = {},
        batch_normalization: bool = True,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.example_input_array = [torch.rand(1, 8), torch.rand(1, 2)]

        self.net = SpaceChargeQuadrupoleMLP(
            num_hidden_layers=num_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            hidden_activation=hidden_activation,
            hidden_activation_args=hidden_activation_args,
            batch_normalization=batch_normalization,
        )

        # Loss to increase influence of rare samples with large deviations from linear
        # beam dynamics
        self.criterion = torch.nn.MSELoss()

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def forward(self, incoming, controls):
        outgoing_deltas = self.net(incoming, controls)
        return outgoing_deltas

    def training_step(self, batch, batch_idx):
        (incoming, controls), true_outgoing_deltas = batch

        predicted_outgoing_deltas = self.net(incoming, controls)

        loss = self.criterion(predicted_outgoing_deltas, true_outgoing_deltas)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (incoming, controls), true_outgoing_deltas = batch

        predicted_outgoing_deltas = self.net(incoming, controls)

        loss = self.criterion(predicted_outgoing_deltas, true_outgoing_deltas)

        self.log("validate/loss", loss, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        (incoming, controls), true_outgoing_deltas = batch

        predicted_outgoing_deltas = self.net(incoming, controls)

        loss = self.criterion(predicted_outgoing_deltas, true_outgoing_deltas)

        self.log("test/loss", loss, sync_dist=True)

        return loss
