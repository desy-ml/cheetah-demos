from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class OcelotSpaceChargeQuadrupoleDataset(Dataset):
    """
    Dataset of beams tracked through quadrupole magnets with space charge using Ocelot.
    In this dataset X is the incoming beam parameters and y is the deltas of the sigmas
    that need to be added to the sigmas of a beam tracked by Cheetah without space
    charge to get the sigmas of a beam tracked by Ocelot with space charge.
    """

    def __init__(
        self,
        stage: Literal["train", "validation", "test"] = "train",
        normalize: bool = False,
        incoming_scaler: Optional[StandardScaler] = None,
        controls_scaler: Optional[StandardScaler] = None,
        outgoing_delta_scaler: Optional[StandardScaler] = None,
    ):
        self.normalize = normalize

        assert stage in ["train", "validation", "test"]

        data_file = Path(__file__).parent / "data" / f"{stage}.yaml"
        with open(data_file, "r") as f:
            sample_dicts = yaml.safe_load(f)

        self.incoming = torch.tensor(
            [
                [
                    sample["incoming"]["sigma_x"],
                    sample["incoming"]["sigma_xp"],
                    sample["incoming"]["sigma_y"],
                    sample["incoming"]["sigma_yp"],
                    sample["incoming"]["sigma_s"],
                    sample["incoming"]["sigma_p"],
                    sample["incoming"]["total_charge"],
                    sample["incoming"]["energy"],
                ]
                for sample in sample_dicts
            ],
            dtype=torch.float32,
        )
        self.controls = torch.tensor(
            [
                [sample["controls"]["length"], sample["controls"]["k1"]]
                for sample in sample_dicts
            ],
            dtype=torch.float32,
        )
        self.outgoing_deltas = torch.tensor(
            [
                [
                    sample["outgoing_deltas"]["sigma_x"],
                    sample["outgoing_deltas"]["sigma_xp"],
                    sample["outgoing_deltas"]["sigma_y"],
                    sample["outgoing_deltas"]["sigma_yp"],
                    sample["outgoing_deltas"]["sigma_s"],
                    sample["outgoing_deltas"]["sigma_p"],
                ]
                for sample in sample_dicts
            ],
            dtype=torch.float32,
        )

        if self.normalize:
            self.setup_normalization(
                incoming_scaler, controls_scaler, outgoing_delta_scaler
            )

    def __len__(self):
        return len(self.incoming)

    def __getitem__(self, index):
        incoming = self.incoming[index]
        controls = self.controls[index]
        outgoing_deltas = self.outgoing_deltas[index]

        if self.normalize:
            incoming = torch.from_numpy(
                self.incoming_scaler.transform([incoming])[0]
            ).float()
            controls = torch.from_numpy(
                self.controls_scaler.transform([controls])[0]
            ).float()
            outgoing_deltas = torch.from_numpy(
                self.outgoing_delta_scaler.transform([outgoing_deltas])[0]
            ).float()

        return (incoming, controls), outgoing_deltas

    def setup_normalization(
        self,
        incoming_scaler: Optional[StandardScaler] = None,
        controls_scaler: Optional[StandardScaler] = None,
        outgoing_delta_scaler: Optional[StandardScaler] = None,
    ) -> None:
        """
        Creates a normalisation scaler for the beam parameters in this dataset. Pass
        already fitted scalers that should be used. If a scaler is not passed, a new one
        is fitted to the data in the dataset.
        """
        self.incoming_scaler = (
            incoming_scaler
            if incoming_scaler is not None
            else StandardScaler().fit(self.incoming)
        )
        self.controls_scaler = (
            controls_scaler
            if controls_scaler is not None
            else StandardScaler().fit(self.controls)
        )
        self.outgoing_delta_scaler = (
            outgoing_delta_scaler
            if outgoing_delta_scaler is not None
            else StandardScaler().fit(self.outgoing_deltas)
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
            incoming_scaler=self.dataset_train.incoming_scaler,
            controls_scaler=self.dataset_train.controls_scaler,
            outgoing_delta_scaler=self.dataset_train.outgoing_delta_scaler,
        )
        self.dataset_test = OcelotSpaceChargeQuadrupoleDataset(
            stage="test",
            normalize=self.normalize,
            incoming_scaler=self.dataset_train.incoming_scaler,
            controls_scaler=self.dataset_train.controls_scaler,
            outgoing_delta_scaler=self.dataset_train.outgoing_delta_scaler,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
