# De-JEPA

This repository is a minimal implementation of **DeJEPA: Non-Contrastive Self-Supervised Learning for Voxel-Level Representations**. It contains only the Python code required for preprocessing, pretraining, linear and nonlinear probing, and full fine-tuning.

## Method

DeJEPA trains a shared 3D feature pyramid on two independently augmented overlapping CT crops. It samples aligned foreground voxels from the overlap, projects their concatenated multi-scale features, and minimizes

```text
L = (1 - lambda) MSE(z, z') + lambda SIGReg(z, z')
```

SIGReg samples random unit directions and applies the Epps-Pulley statistic to the one-dimensional embedding projections. The defaults match the paper:

| Setting | Value |
|---|---:|
| Crop | 32 x 128 x 128 voxels |
| Patients per batch | 10 |
| Aligned voxels per patient | 1,024 |
| Projection dimension | 128 |
| SIGReg directions | 1,024 |
| SIGReg weight | 0.001 |
| Pretraining steps | 100,000 |
| Optimizer | AdamW |
| Learning rate | 8e-4 |
| Weight decay | 5e-4 |
| Warmup | 5% |
| Schedule | Cosine |

## Installation

Python 3.11 and a CUDA GPU are recommended. The paper experiments used one A100 GPU.

```bash
pip install -r requirements.txt
```

## Data

No data is distributed here. The pretraining corpus in the paper combines FLARE'23, TotalSegmentator, LIDC-IDRI, LUNA16, STOIC'21, TCIA COVID-19, and HNSCC after corrupt-volume filtering, with BTCV excluded. Each source volume needs a binary body mask named `<case>_body_mask.nii.gz`.

Prepare each image and mask directory into the random-access format used by training:

```bash
python prepare_data.py \
  --images /path/to/images \
  --masks /path/to/body_masks \
  --output /path/to/pretrain_cache
```

The command resamples to 1 x 1 x 2 mm, dilates and crops to the body mask, converts to RAS, and writes a compressed volume plus its foreground-coordinate index. Multiple prepared datasets can be combined by placing their `.b2nd` files in one directory with unique case names.

BTCV and MSD evaluation directories use this layout:

```text
dataset/
  imagesTr/
  labelsTr/
  bodymasksTr/
```

Each split is a JSON file with digit-only subject identifiers:

```json
{
  "train": ["0001", "0002"],
  "val": ["0003"],
  "test": ["0004"]
}
```

Use the five published BTCV folds and three published MSD folds to compare directly with the paper. Few-shot experiments use separate 1-, 5-, and 10-case training splits and are repeated over three splits.

## Pretraining

```bash
python train.py \
  --stage pretrain \
  --data /path/to/pretrain_cache \
  --output runs/pretrain
```

The final checkpoint is `runs/pretrain/final.ckpt`. Resume an interrupted run with `--resume /path/to/last.ckpt`.

## Probing

The probe command trains the linear 13k-parameter and nonlinear 1M-parameter heads together while keeping the FPN frozen. Two loader items each produce four crops, giving the downstream batch size of eight used in the paper. The default 45,000 steps, AdamW learning rate of `8e-4`, zero weight decay, 5% warmup, and cosine decay match the reported protocol.

```bash
python train.py \
  --stage probe \
  --checkpoint runs/pretrain/final.ckpt \
  --data /path/to/BTCV \
  --split /path/to/splits/split_0.json \
  --output runs/btcv/probe/fold_0
```

For an MSD task, set its foreground class count. Use `2` for Task03 Liver and Task08 Hepatic Vessel, and `1` for Task06 Lung and Task09 Spleen.

```bash
python train.py \
  --stage probe \
  --checkpoint runs/pretrain/final.ckpt \
  --data /path/to/Task03_Liver \
  --split /path/to/Task03_Liver/split_0.json \
  --classes 2 \
  --output runs/msd/task03/probe/fold_0
```

## Fine-tuning

Fine-tuning uses 90,000 steps, AdamW with learning rate `3e-4` and weight decay `1e-5`. The pretrained backbone stays frozen for 30,000 steps and then ramps to the head learning rate.

```bash
python train.py \
  --stage finetune \
  --checkpoint runs/pretrain/final.ckpt \
  --data /path/to/BTCV \
  --split /path/to/splits/split_0.json \
  --output runs/btcv/finetune/fold_0
```

Omit `--checkpoint` for the from-scratch baseline. Add `--augmentation msd` for MSD fine-tuning.

## Expected results

With the paper corpus, preprocessing, folds, and hardware, the reported BTCV mean Dice is 66.4 for linear probing, 74.1 for nonlinear probing, and 80.7 for fine-tuning. The 1-shot BTCV result is 46.6 for linear probing and 43.4 for fine-tuning. Exact values remain subject to dataset versions, preprocessing, CUDA kernels, and random seeds.
