from collections.abc import Callable, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from transformers import get_cosine_schedule_with_warmup

from model import FPN3D, LinearHead, NonlinearHead, Projector, select_from_pyramid
from objective import DeJEPALoss


def targets_from_labels(labels: torch.Tensor, classes: int) -> torch.Tensor:
    labels = labels.squeeze(1).long()
    one_hot = F.one_hot(labels, num_classes=classes + 1)
    return one_hot.permute(0, 4, 1, 2, 3)[:, 1:].float()


def segmentation_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    elementwise = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    probabilities = torch.sigmoid(logits.float())
    if mask is None:
        binary = elementwise.mean()
    else:
        valid = mask.expand_as(elementwise) > 0
        binary = elementwise[valid].mean()
        probabilities = torch.where(
            valid, probabilities, torch.zeros_like(probabilities)
        )
        targets = torch.where(valid, targets, torch.zeros_like(targets))
    intersection = (probabilities * targets).sum((2, 3, 4), dtype=torch.float32)
    denominator = (probabilities + targets).sum((2, 3, 4), dtype=torch.float32)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return binary + (1.0 - dice).mean()


def sliding_boxes(
    image_size: Sequence[int], patch_size: Sequence[int], overlap_fraction: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    image_size = np.asarray(image_size)
    patch_size = np.asarray(patch_size)
    overlap = np.ceil(patch_size * overlap_fraction).astype(int)
    overlap += overlap % 2
    stride = patch_size - overlap
    boxes = []
    grid = tuple(1 + np.ceil((image_size - patch_size) / stride).astype(int))
    for location in np.ndindex(grid):
        start = np.asarray(location) * stride
        stop = np.minimum(start + patch_size, image_size)
        start = stop - patch_size
        boxes.append((start, stop))
    return boxes


def tiled_predict(
    images: torch.Tensor,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    patch_size: tuple[int, int, int] = (32, 128, 128),
    overlap_fraction: float = 0.5,
) -> torch.Tensor:
    image_size = np.asarray(images.shape[-3:])
    patch = np.asarray(patch_size)
    overlap = np.ceil(patch * overlap_fraction).astype(int)
    overlap += overlap % 2
    output = None
    for start, stop in sliding_boxes(image_size, patch, overlap_fraction):
        prediction = predictor(images[(..., *map(slice, start, stop))])
        if output is None:
            output = torch.zeros(
                images.shape[0],
                prediction.shape[1],
                *image_size,
                device=images.device,
                dtype=prediction.dtype,
            )
        inner_start = np.where(start > 0, start + overlap // 2, 0)
        inner_stop = np.where(stop < image_size, stop - overlap // 2, image_size)
        output[(..., *map(slice, inner_start, inner_stop))] = prediction[
            (..., *map(slice, inner_start - start, inner_stop - start))
        ]
    return output


def load_backbone(backbone: FPN3D, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    prefixes = ("backbone.", "net.")
    selected = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                selected[key.removeprefix(prefix)] = value
                break
    if not selected:
        selected = state
    missing, unexpected = backbone.load_state_dict(selected, strict=False)
    if missing:
        raise RuntimeError(f"missing backbone parameters: {missing}")
    backbone_keys = set(backbone.state_dict())
    unexpected_backbone = [key for key in unexpected if key in backbone_keys]
    if unexpected_backbone:
        raise RuntimeError(f"unexpected backbone parameters: {unexpected_backbone}")


class PretrainModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 8e-4,
        weight_decay: float = 5e-4,
        steps: int = 100000,
        sigreg_weight: float = 0.001,
        projections: int = 1024,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = FPN3D()
        self.projector = Projector(self.backbone.embedding_size, output_dim=128)
        self.objective = DeJEPALoss(
            sigreg_weight=sigreg_weight, projections=projections
        )

    def training_step(self, batch, batch_index):
        first, second, first_coordinates, second_coordinates = batch
        first_features = select_from_pyramid(
            self.backbone(first), first_coordinates
        ).flatten(0, 1)
        second_features = select_from_pyramid(
            self.backbone(second), second_coordinates
        ).flatten(0, 1)
        losses = self.objective(
            self.projector(first_features), self.projector(second_features)
        )
        self.log_dict(
            {f"train/{name}": value for name, value in losses.items()}, prog_bar=True
        )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.projector.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.steps // 20,
            num_training_steps=self.hparams.steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ProbeModule(pl.LightningModule):
    def __init__(
        self, classes: int = 13, learning_rate: float = 8e-4, normalize: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = FPN3D()
        self.linear = LinearHead(self.backbone.channels, classes, normalize=normalize)
        self.nonlinear = NonlinearHead(self.backbone.channels, classes)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.validation_linear = DiceMetric(reduction="mean_batch")
        self.validation_nonlinear = DiceMetric(reduction="mean_batch")
        self.test_linear = DiceMetric(reduction="mean_batch", ignore_empty=False)
        self.test_nonlinear = DiceMetric(reduction="mean_batch", ignore_empty=False)

    def training_step(self, batch, batch_index):
        targets = targets_from_labels(batch["label"], self.hparams.classes)
        self.backbone.eval()
        with torch.no_grad():
            pyramid = self.backbone(batch["image"])
        linear = segmentation_loss(self.linear(pyramid), targets, batch.get("mask"))
        nonlinear = segmentation_loss(
            self.nonlinear(pyramid), targets, batch.get("mask")
        )
        self.log_dict({"train/linear_loss": linear, "train/nonlinear_loss": nonlinear})
        return linear + nonlinear

    def _evaluate(self, batch, stage: str) -> None:
        targets = targets_from_labels(batch["label"], self.hparams.classes).int()
        images = batch["image"]
        with torch.no_grad():
            linear = tiled_predict(
                images, lambda x: torch.sigmoid(self.linear(self.backbone(x)))
            )
            nonlinear = tiled_predict(
                images, lambda x: torch.sigmoid(self.nonlinear(self.backbone(x)))
            )
        if "mask" in batch:
            valid = batch["mask"].expand_as(linear) > 0
            linear = torch.where(valid, linear, torch.zeros_like(linear))
            nonlinear = torch.where(valid, nonlinear, torch.zeros_like(nonlinear))
        linear = decollate_batch((linear > 0.5).int())
        nonlinear = decollate_batch((nonlinear > 0.5).int())
        targets = decollate_batch(targets)
        if stage == "val":
            self.validation_linear(linear, targets)
            self.validation_nonlinear(nonlinear, targets)
        else:
            self.test_linear(linear, targets)
            self.test_nonlinear(nonlinear, targets)

    def validation_step(self, batch, batch_index):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_index):
        self._evaluate(batch, "test")

    def _finish_metrics(
        self, linear: DiceMetric, nonlinear: DiceMetric, stage: str
    ) -> None:
        linear_scores = linear.aggregate()
        nonlinear_scores = nonlinear.aggregate()
        linear.reset()
        nonlinear.reset()
        self.log(f"{stage}/linear_dice", linear_scores.mean())
        self.log(f"{stage}/nonlinear_dice", nonlinear_scores.mean())
        for index, score in enumerate(linear_scores):
            self.log(f"{stage}/linear_class_{index + 1}", score)
            self.log(f"{stage}/nonlinear_class_{index + 1}", nonlinear_scores[index])

    def on_validation_epoch_end(self) -> None:
        self._finish_metrics(self.validation_linear, self.validation_nonlinear, "val")

    def on_test_epoch_end(self) -> None:
        self._finish_metrics(self.test_linear, self.test_nonlinear, "test")

    def configure_optimizers(self):
        parameters = list(self.linear.parameters()) + list(self.nonlinear.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=self.hparams.learning_rate, weight_decay=0.0
        )
        steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(optimizer, steps // 20, steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class FinetuneModule(pl.LightningModule):
    def __init__(self, classes: int = 13, learning_rate: float = 3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = FPN3D()
        self.head = LinearHead(self.backbone.channels, classes)
        self.validation_metric = DiceMetric(reduction="mean_batch")
        self.test_metric = DiceMetric(reduction="mean_batch", ignore_empty=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def training_step(self, batch, batch_index):
        targets = targets_from_labels(batch["label"], self.hparams.classes)
        loss = segmentation_loss(self(batch["image"]), targets, batch.get("mask"))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _evaluate(self, batch, stage: str) -> None:
        targets = targets_from_labels(batch["label"], self.hparams.classes).int()
        probabilities = sliding_window_inference(
            batch["image"],
            roi_size=(32, 128, 128),
            sw_batch_size=4,
            predictor=lambda x: torch.sigmoid(self(x)),
            overlap=0.5,
            mode="gaussian",
        )
        if "mask" in batch:
            valid = batch["mask"].expand_as(probabilities) > 0
            probabilities = torch.where(
                valid, probabilities, torch.zeros_like(probabilities)
            )
        predictions = decollate_batch((probabilities > 0.5).int())
        targets = decollate_batch(targets)
        metric = self.validation_metric if stage == "val" else self.test_metric
        metric(predictions, targets)

    def validation_step(self, batch, batch_index):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_index):
        self._evaluate(batch, "test")

    def _finish_metric(self, metric: DiceMetric, stage: str) -> None:
        scores = metric.aggregate()
        metric.reset()
        self.log(f"{stage}/dice", scores.mean())
        for index, score in enumerate(scores):
            self.log(f"{stage}/class_{index + 1}", score)

    def on_validation_epoch_end(self) -> None:
        self._finish_metric(self.validation_metric, "val")

    def on_test_epoch_end(self) -> None:
        self._finish_metric(self.test_metric, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )
        steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(optimizer, steps // 20, steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
