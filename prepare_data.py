import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk


def resample(
    image: sitk.Image, spacing: tuple[float, float, float], label: bool
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    output_size = [
        int(round(size * source / target))
        for size, source, target in zip(original_size, original_spacing, spacing)
    ]
    operation = sitk.ResampleImageFilter()
    operation.SetOutputSpacing(spacing)
    operation.SetSize(output_size)
    operation.SetOutputDirection(image.GetDirection())
    operation.SetOutputOrigin(image.GetOrigin())
    operation.SetTransform(sitk.Transform())
    operation.SetDefaultPixelValue(0)
    operation.SetInterpolator(sitk.sitkNearestNeighbor if label else sitk.sitkLinear)
    return operation.Execute(image)


def convert_case(arguments: tuple[str, str, str, tuple[float, float, float]]) -> str:
    image_path, mask_path, output_path, spacing = arguments
    image = resample(sitk.ReadImage(image_path), spacing, False)
    mask = resample(sitk.ReadImage(mask_path), spacing, True)
    mask = sitk.BinaryThreshold(mask, 1, 255, 1, 0)
    mask = sitk.BinaryDilate(mask, [6, 6, 3], sitk.sitkBall)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask)
    if 1 not in statistics.GetLabels():
        raise RuntimeError(f"empty body mask: {mask_path}")
    box = statistics.GetBoundingBox(1)
    image = sitk.RegionOfInterest(image, box[3:], box[:3])
    mask = sitk.RegionOfInterest(mask, box[3:], box[:3])
    image = sitk.DICOMOrient(image, "RAS")
    mask = sitk.DICOMOrient(mask, "RAS")
    image_array = sitk.GetArrayFromImage(image).astype(np.int16)
    mask_array = sitk.GetArrayFromImage(mask).astype(np.int16)
    if any(size < minimum for size, minimum in zip(image_array.shape, (32, 128, 128))):
        raise RuntimeError(f"volume is smaller than a training patch: {image_path}")
    output = Path(output_path)
    compressed = np.stack((image_array, mask_array))
    blosc2.asarray(
        compressed,
        urlpath=str(output),
        mode="w",
        cparams={"codec": blosc2.Codec.LZ4, "clevel": 5},
    )
    anchors = np.argwhere(mask_array > 0).astype(np.uint16)
    blosc2.asarray(
        anchors,
        urlpath=str(output.with_name(f"{output.stem}_anchors.b2nd")),
        mode="w",
        cparams={
            "codec": blosc2.Codec.ZSTD,
            "clevel": 5,
            "filters": [blosc2.Filter.BITSHUFFLE],
        },
    )
    return output.name


def prepare_dataset(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    workers: int = 8,
) -> None:
    images = sorted(Path(image_dir).glob("*.nii.gz"))
    if not images:
        raise FileNotFoundError(f"no NIfTI files found in {image_dir}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for image in images:
        stem = image.name.removesuffix(".nii.gz")
        mask = Path(mask_dir) / f"{stem}_body_mask.nii.gz"
        if not mask.exists():
            raise FileNotFoundError(mask)
        jobs.append((str(image), str(mask), str(output / f"{stem}.b2nd"), spacing))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name in pool.map(convert_case, jobs):
            print(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 2.0))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare_dataset(
        args.images, args.masks, args.output, tuple(args.spacing), args.workers
    )


if __name__ == "__main__":
    main()
