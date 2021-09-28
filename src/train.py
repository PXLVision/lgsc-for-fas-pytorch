from argparse import ArgumentParser, Namespace

import safitty
import pytorch_lightning as pl
import mlflow
import os

from pl_model import LightningModel
from models.scan import SCAN

def init_mlflow():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://s3.ewstorage.ch'
    mlflow.set_tracking_uri('https://mlflow.pxl-vision.com:9056')
    experiment = mlflow.get_experiment_by_name('liveness-anomaly')
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            'liveness-anomaly', artifact_location='s3://pxl-ew-s3-dev-ml-mlflow/liveness-anomaly/pytorchliveness'
        )
    else:
        experiment_id = experiment.experiment_id

    return experiment_id



if __name__ == "__main__":
    experiment_id = init_mlflow()
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    model = LightningModel(hparams=configs)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="total_val_loss")
    trainer = pl.Trainer.from_argparse_args(
        configs,
        gpus=-1,
        accelerator="ddp",
        fast_dev_run=False,
        callbacks=[checkpoint_callback],
        default_root_dir=configs.default_root_dir,
    )
    mlflow.autolog()
    with mlflow.start_run(run_name=f"liveness-anomaly", experiment_id=experiment_id):
        trainer.fit(model)
