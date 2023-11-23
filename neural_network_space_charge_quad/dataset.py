from typing import Literal, Optional

import lightning as L
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class OcelotSpaceChargeQuadrupoleDataset(Dataset):
    """
    Dataset of beams tracked through quadrupole magnets with space charge using Ocelot.
    In this dataset X is the incoming beam parameters and y is the outgoing beam
    parameters.

    This dataset reads in the original Ocelot beam files and outputs the beam parameters
    as they are understoof by Cheetah in the following order:
     - mu_x
     - mu_xp
     - mu_y
     - mu_yp
     - sigma_x
     - sigma_xp
     - sigma_y
     - sigma_yp
     - sigma_s
     - sigma_p
     - cor_x
     - cor_y
     - cor_s
     - energy
     - total_charge
    """

    def __init__(
        self,
        stage: Literal["train", "validation", "test"] = "train",
        normalize: bool = False,
        scaler: Optional[StandardScaler] = None,
    ):
        self.normalize = normalize

        assert stage in ["train", "validation", "test"]

        # TODO: Read Ocelot beams (incoming and outgoing)
        # TODO: Convert to `cheetah.ParameterBeam`s (incoming and outgoing)
        # TODO: Keep only the beam parameters of the Cheetah beams (incoming and
        # outgoing)

        if self.normalize:
            self.setup_normalization(scaler)

    def __len__(self):
        return len(self.incoming_beam_parameters)

    def __getitem__(self, index):
        rf_settings = self.rf_settings[index]
        formfactor = self.formfactors[index]
        current_profile = self.current_profiles[index]
        bunch_length = self.bunch_lengths[index]

        if self.normalize:
            rf_settings = self.rf_scaler.transform([rf_settings])[0]
            formfactor = self.formfactor_scaler.transform([formfactor])[0]
            current_profile = self.current_scaler.transform([current_profile])[0]
            bunch_length = self.bunch_length_scaler.transform([bunch_length])[0]

        rf_settings = torch.tensor(rf_settings, dtype=torch.float32)
        formfactor = torch.tensor(formfactor, dtype=torch.float32)
        current_profile = torch.tensor(current_profile, dtype=torch.float32)
        bunch_length = torch.tensor(bunch_length, dtype=torch.float32)

        return (rf_settings, formfactor), (current_profile, bunch_length)

    def setup_normalization(self, scaler: Optional[StandardScaler] = None) -> None:
        """
        Creates a normalisation scaler for the beam parameters in this dataset. Pass
        already fitted scalers that should be used. If a scaler is not passed, a new one
        is fitted to the data in the dataset.
        """
        self.scaler = (
            scaler
            if scaler is not None
            else StandardScaler().fit(
                torch.concatenate(
                    [self.incoming_beam_parameters, self.outgoing_beam_parameters],
                    dim=0,
                )
            )
        )


class OcelotSpaceChargeQuadrupoleDataModule(L.LightningDataModule):
    """
    Data Module for beams tracked through quadrupole magnets with space charge using
    Ocelot.
    """

    def __init__(self, batch_size=32, normalize=False, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset_train = OcelotSpaceChargeQuadrupoleDataset(
            stage="train", normalize=self.normalize
        )
        self.dataset_val = OcelotSpaceChargeQuadrupoleDataset(
            stage="validation",
            normalize=self.normalize,
            scaler=self.dataset_train.scaler,
        )
        self.dataset_test = OcelotSpaceChargeQuadrupoleDataset(
            stage="test", normalize=self.normalize, scaler=self.dataset_train.scaler
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
