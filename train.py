import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from data import InterPatientBatchSampler, PretrainDataset, SegmentationDataModule
from modules import FinetuneModule, PretrainModule, ProbeModule, load_backbone


def make_trainer(
    output: Path,
    steps: int,
    epochs: int,
    callbacks: list,
    precision: str,
    gradient_clip: float | None = None,
) -> pl.Trainer:
    output.mkdir(parents=True, exist_ok=True)
    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        max_steps=steps,
        max_epochs=epochs,
        benchmark=True,
        gradient_clip_val=gradient_clip,
        callbacks=callbacks,
        logger=CSVLogger(output, name="logs"),
        default_root_dir=output,
    )


def segmentation_data(
    args: argparse.Namespace, augmentation: str
) -> SegmentationDataModule:
    if args.split is None:
        raise ValueError("--split is required for probing and fine-tuning")
    return SegmentationDataModule(
        root=args.data,
        split=args.split,
        classes=args.classes,
        batch_size=2,
        workers=args.workers,
        augmentation=augmentation,
        cache_dir=args.cache,
        spacing=tuple(args.spacing),
        window=tuple(args.window),
    )


def pretrain(args: argparse.Namespace) -> None:
    steps = 100000 if args.steps is None else args.steps
    learning_rate = 8e-4 if args.learning_rate is None else args.learning_rate
    dataset = PretrainDataset(args.data)
    sampler = InterPatientBatchSampler(
        len(dataset), args.batch_size, batches=250, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = PretrainModule(
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        steps=steps,
        sigreg_weight=args.sigreg_weight,
        projections=args.projections,
    )
    output = Path(args.output)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", save_last=True, every_n_train_steps=5000
    )
    trainer = make_trainer(output, steps, steps // 250, [checkpoint], "16-mixed")
    trainer.fit(model, train_dataloaders=loader, ckpt_path=args.resume)
    trainer.save_checkpoint(output / "final.ckpt")


def probe(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for probing")
    steps = 45000 if args.steps is None else args.steps
    learning_rate = 8e-4 if args.learning_rate is None else args.learning_rate
    model = ProbeModule(
        classes=args.classes, learning_rate=learning_rate, normalize=args.normalize
    )
    load_backbone(model.backbone, args.checkpoint)
    data = segmentation_data(args, args.augmentation or "simple")
    output = Path(args.output)
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", save_last=True)
    trainer = make_trainer(output, steps, steps // 300, [checkpoint], "bf16-mixed")
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint(output / "final.ckpt")
    trainer.test(model, datamodule=data)


def finetune(args: argparse.Namespace) -> None:
    steps = 90000 if args.steps is None else args.steps
    learning_rate = 3e-4 if args.learning_rate is None else args.learning_rate
    model = FinetuneModule(classes=args.classes, learning_rate=learning_rate)
    callbacks = []
    if args.checkpoint is not None:
        load_backbone(model.backbone, args.checkpoint)
        callbacks.append(
            BackboneFinetuning(unfreeze_backbone_at_epoch=args.freeze_steps // 300)
        )
    output = Path(args.output)
    callbacks.append(ModelCheckpoint(dirpath=output / "checkpoints", save_last=True))
    data = segmentation_data(args, args.augmentation or "heavy")
    trainer = make_trainer(output, steps, steps // 300, callbacks, "bf16-mixed", 1.0)
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint(output / "final.ckpt")
    trainer.test(model, datamodule=data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("pretrain", "probe", "finetune")
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--classes", type=int, default=13)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sigreg-weight", type=float, default=0.001)
    parser.add_argument("--projections", type=int, default=1024)
    parser.add_argument("--freeze-steps", type=int, default=30000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--augmentation", choices=("simple", "heavy", "msd"))
    parser.add_argument("--cache")
    parser.add_argument("--spacing", type=float, nargs=3, default=(2.0, 1.0, 1.0))
    parser.add_argument("--window", type=float, nargs=2, default=(-1350.0, 1000.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    {"pretrain": pretrain, "probe": probe, "finetune": finetune}[args.stage](args)


if __name__ == "__main__":
    main()
