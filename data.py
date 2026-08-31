import json
import re
from glob import glob
from pathlib import Path

import blosc2
import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    OneOf,
    Orientationd,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.utils.data import BatchSampler, RandomSampler


class RandomCTWindow(MapTransform):
    def __init__(
        self,
        keys: list[str],
        inner: tuple[float, float] = (-1000.0, 300.0),
        outer: tuple[float, float] = (-1350.0, 1000.0),
        probability: float = 0.8,
    ) -> None:
        super().__init__(keys)
        self.inner = inner
        self.outer = outer
        self.probability = probability

    def __call__(self, data: dict) -> dict:
        output = dict(data)
        if np.random.random() < self.probability:
            low = np.random.uniform(self.outer[0], self.inner[0])
            high = np.random.uniform(self.inner[1], self.outer[1])
            for key in self.keys:
                output[key] = np.clip((output[key] - low) / (high - low), 0.0, 1.0)
        return output


def pretrain_transforms(
    inner_window: tuple[float, float] = (-1000.0, 300.0),
    outer_window: tuple[float, float] = (-1350.0, 1000.0),
    noise_std: float = 45.0,
) -> Compose:
    return Compose(
        [
            OneOf(
                [
                    RandGaussianSmoothd(keys=["image"], prob=0.5),
                    RandGaussianSharpend(keys=["image"], prob=0.5),
                ]
            ),
            RandGaussianNoised(
                keys=["image"], mean=0.0, std=noise_std, prob=0.5, sample_std=True
            ),
            OneOf(
                [
                    RandomCTWindow(
                        ["image"], inner_window, outer_window, probability=1.0
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=outer_window[0],
                        a_max=outer_window[1],
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                ],
                weights=[0.8, 0.2],
            ),
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.5),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        patch_size: tuple[int, int, int] = (32, 128, 128),
        points: int = 1024,
    ) -> None:
        self.files = sorted(
            path
            for path in Path(root).glob("*.b2nd")
            if not path.name.endswith("_anchors.b2nd")
        )
        if not self.files:
            raise FileNotFoundError(f"no .b2nd volumes found in {root}")
        missing = [
            path
            for path in self.files
            if not path.with_name(f"{path.stem}_anchors.b2nd").exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing anchor cache for {missing[0]}")
        self.patch_size = np.asarray(patch_size)
        self.points = points
        self.minimum_overlap = np.cbrt(points / np.prod(self.patch_size))
        self.transform = pretrain_transforms()

    def __len__(self) -> int:
        return len(self.files)

    def _box(
        self,
        shape: tuple[int, int, int],
        anchor: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        shape = np.asarray(shape)
        low = np.maximum(0, anchor - self.patch_size)
        high = np.minimum(shape - self.patch_size, anchor)
        if reference is not None:
            overlap = (self.patch_size * self.minimum_overlap).astype(int)
            low = np.maximum(low, reference[:, 0] - self.patch_size + overlap)
            high = np.minimum(high, reference[:, 0] + self.patch_size - overlap)
        high = np.maximum(low, high)
        start = np.asarray(
            [np.random.randint(a, b + 1) if b > a else a for a, b in zip(low, high)]
        )
        return np.stack((start, start + self.patch_size), axis=1)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.files[index]
        anchors = blosc2.open(
            str(path.with_name(f"{path.stem}_anchors.b2nd")), mode="r"
        )
        anchor = anchors[np.random.randint(len(anchors))]
        volume = blosc2.open(str(path), mode="r")
        first_box = self._box(volume.shape[1:], anchor)
        second_box = self._box(volume.shape[1:], anchor, first_box)
        first_slice = (0,) + tuple(slice(start, stop) for start, stop in first_box)
        second_slice = (0,) + tuple(slice(start, stop) for start, stop in second_box)
        first_patch = volume[first_slice]
        second_patch = volume[second_slice]
        overlap_start = np.maximum(first_box[:, 0], second_box[:, 0])
        overlap_stop = np.minimum(first_box[:, 1], second_box[:, 1])
        mask_slice = (1,) + tuple(
            slice(start, stop) for start, stop in zip(overlap_start, overlap_stop)
        )
        foreground = np.argwhere(volume[mask_slice] > 0)
        if len(foreground) == 0:
            raise RuntimeError(f"empty overlap foreground in {path}")
        global_coordinates = foreground + overlap_start
        replacement = len(global_coordinates) < self.points
        selected = np.random.choice(
            len(global_coordinates), self.points, replace=replacement
        )
        first_coordinates = global_coordinates[selected] - first_box[:, 0]
        second_coordinates = global_coordinates[selected] - second_box[:, 0]
        first_patch = self.transform({"image": first_patch})["image"].unsqueeze(0)
        second_patch = self.transform({"image": second_patch})["image"].unsqueeze(0)
        return (
            first_patch.float(),
            second_patch.float(),
            torch.as_tensor(first_coordinates, dtype=torch.long),
            torch.as_tensor(second_coordinates, dtype=torch.long),
        )


class InterPatientBatchSampler(BatchSampler):
    def __init__(
        self, patients: int, batch_size: int, batches: int = 250, seed: int = 42
    ) -> None:
        if patients < batch_size:
            raise ValueError("the number of patients must be at least the batch size")
        self.patients = patients
        self.batch_size = batch_size
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            yield generator.choice(
                self.patients, self.batch_size, replace=False
            ).tolist()

    def __len__(self) -> int:
        return self.batches


def subject_id(path: str) -> str:
    return re.sub(r"\D", "", Path(path).name.replace(".nii.gz", ""))


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        split: str | Path,
        classes: int = 13,
        batch_size: int = 2,
        workers: int = 8,
        augmentation: str = "simple",
        cache_dir: str | Path | None = None,
        spacing: tuple[float, float, float] = (2.0, 1.0, 1.0),
        window: tuple[float, float] = (-1350.0, 1000.0),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = Path(split)
        self.classes = classes
        self.batch_size = batch_size
        self.workers = workers
        self.augmentation = augmentation
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.spacing = spacing
        self.window = window

    def setup(self, stage: str | None = None) -> None:
        images = sorted(glob(str(self.root / "imagesTr" / "*.nii.gz")))
        labels = sorted(glob(str(self.root / "labelsTr" / "*.nii.gz")))
        if len(images) != len(labels) or not images:
            raise RuntimeError(
                "imagesTr and labelsTr must contain matching NIfTI files"
            )
        records = []
        for image, label in zip(images, labels):
            stem = Path(image).name.replace(".nii.gz", "")
            mask = self.root / "bodymasksTr" / f"{stem}_body_mask.nii.gz"
            if not mask.exists():
                raise FileNotFoundError(mask)
            records.append({"image": image, "label": label, "mask": str(mask)})
        partitions = json.loads(self.split.read_text())
        self.records = {
            name: [
                record
                for record in records
                if subject_id(record["image"]) in set(partitions.get(name, []))
            ]
            for name in ("train", "val", "test")
        }

    def _transforms(self, training: bool) -> Compose:
        keys = ["image", "label", "mask"]
        transforms = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="SAR"),
            Spacingd(
                keys=keys, pixdim=self.spacing, mode=("bilinear", "nearest", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window[0],
                a_max=self.window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=keys, source_key="mask"),
        ]
        if training:
            transforms.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key="label",
                    spatial_size=(32, 128, 128),
                    ratios=[self.classes] + [1] * self.classes,
                    num_classes=self.classes + 1,
                    num_samples=4,
                )
            )
            if self.augmentation in {"heavy", "msd"}:
                transforms.extend(
                    [
                        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                    ]
                )
            if self.augmentation == "msd":
                transforms.extend(
                    [
                        RandFlipd(keys=keys, prob=0.5),
                        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
                    ]
                )
        transforms.append(EnsureTyped(keys=keys))
        return Compose(transforms)

    def _dataset(self, subset: str, training: bool):
        transform = self._transforms(training)
        if self.cache_dir is None:
            return Dataset(self.records[subset], transform)
        directory = self.cache_dir / subset
        directory.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(self.records[subset], transform, cache_dir=directory)

    def train_dataloader(self) -> DataLoader:
        dataset = self._dataset("train", True)
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=300 * self.batch_size
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("val", False),
            batch_size=1,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("test", False),
            batch_size=1,
            num_workers=self.workers,
            pin_memory=True,
        )
