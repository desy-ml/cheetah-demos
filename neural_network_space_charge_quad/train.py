import time

from dataset import OcelotSpaceChargeQuadrupoleDataModule
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from models import SupervisedSpaceChargeQuadrupoleInference


def main():
    config = {
        "batch_normalization": True,
        "batch_size": 256,
        "hidden_activation": "Softplus",
        "hidden_layer_width": 128,
        "learning_rate": 0.006883835325349274,
        "max_epochs": 10_000,
        "num_hidden_layers": 4,
        "use_logarithm": True,
    }

    wandb_logger = WandbLogger(project="space-charge-quadrupole", config=config)
    config = dict(wandb_logger.experiment.config)

    data_module = OcelotSpaceChargeQuadrupoleDataModule(
        batch_size=config["batch_size"],
        num_workers=10,
        use_logarithm=config["use_logarithm"],
        normalize=True,
    )
    model = SupervisedSpaceChargeQuadrupoleInference(
        batch_normalization=config["batch_normalization"],
        hidden_activation=config["hidden_activation"],
        hidden_layer_width=config["hidden_layer_width"],
        learning_rate=config["learning_rate"],
        num_hidden_layers=config["num_hidden_layers"],
    )

    early_stopping_callback = EarlyStopping(
        monitor="validate/loss", mode="min", patience=10
    )

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        callbacks=early_stopping_callback,
    )
    trainer.fit(model, data_module)

    time.sleep(10)


if __name__ == "__main__":
    main()
