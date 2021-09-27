from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import safitty
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pl_model import LightningModel
from datasets import get_test_augmentations, Dataset
from outputs import create_graphs_and_metrics


def prepare_infer_dataloader(args: Namespace) -> DataLoader:
    transforms = get_test_augmentations(args.image_size)
    df = pd.read_csv(args.infer_df)
    dataset = Dataset(
        df, args.root, transforms, None, args.with_labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return dataloader


def load_model_from_checkpoint(checkpoints: str, device: str) -> LightningModel:
    model = LightningModel.load_from_checkpoint(checkpoints)
    model.eval()
    model.to(device)
    return model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    return Namespace(**configs)


def infer_model(
    model: LightningModel,
    dataloader: DataLoader,
    device: str = "cuda:0",
    verbose: bool = False,
    with_labels: bool = True,
    log_folder: Path = ""
) -> Union[Tuple[float, float, float, float, float], List[float]]:
    scores = []
    targets = torch.Tensor()
    output_df = pd.DataFrame(columns=["Score", "Label", "Image"])
    with torch.no_grad():
        model.eval()
        if verbose:
            dataloader = tqdm(dataloader)
        for batch in dataloader:
            if with_labels:
                images, labels, image_path = batch
                labels = labels.float()
                images = images.to(device)
            else:
                images = batch.to(device)
            cues = model(images)

            for i in range(cues.shape[0]):
                score = 1.0 - cues[i, ...].mean().cpu()
                scores.append(score)
                show_cue(images[i].cpu(), cues[i].cpu(), Path(image_path[i]), Path(log_folder))
                result_list = [score, int(labels[i]), image_path[i]]
                result_series = pd.Series(result_list, index=output_df.columns)
                output_df = output_df.append(result_series, ignore_index=True)
            if with_labels:
                targets = torch.cat([targets, labels])
    if with_labels:
        create_graphs_and_metrics(output_df, log_folder)
        return
    else:
        return scores


def show_cue(imgs, cue, img_path, log_folder):
    c, h, w = imgs.shape
    cues = np.zeros((h,  w * 2, 3), dtype=np.uint8)
    img = imgs.numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    cc = cue.numpy().transpose(1, 2, 0)
    cc = (cc - cc.min()) / (cc.max() - cc.min()) * 255
    cc = cc.astype(np.uint8)
    cues[:, 0:w, :] = img.astype(np.uint8)
    cues[:, w:, :] = cc.astype(np.uint8)
    cue_dir = log_folder / "cues" / img_path.parent.name
    if not cue_dir.exists():
        cue_dir.mkdir(parents=True)
    img_name = img_path.name
    cv2.cvtColor(cues, cv2.COLOR_RGB2BGR, cues)
    cv2.imwrite(str(cue_dir / img_name), cues)

if __name__ == "__main__":
    args_ = parse_args()
    model_ = load_model_from_checkpoint(args_.checkpoints, args_.device)

    dataloader_ = prepare_infer_dataloader(args_)

    if args_.with_labels:
        infer_model(model_, dataloader_, args_.device, args_.verbose, True, Path(args_.out_folder))
    else:
        scores_ = infer_model(model_, dataloader_, args_.device, False, False)
        # if you don't have answers you can write your scores into some file
        with open(args_.out_file, "w") as file:
            file.write("\n".join(list(map(lambda x: str(x), scores_))))
