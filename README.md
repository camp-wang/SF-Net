# SF-PINet / SF-Net

Code accompanying the manuscript “一种具有空间频率感知与质量引导自精炼的半监督夜间图像去雾方法”.

## Repository contents

- `model/test1.py`: SF-Net implementation (`Net1`). SFDA blocks are used in the encoder and decoder; DSRM blocks are used in the bottleneck.
- `train_1.py`: supervised training on paired synthetic data.
- `self_refinement_nhm.py`: teacher–student self-refinement with uncertainty-filtered pseudo-labels, EMA updates, and quality-based segment rollback.
- `train_unreal_train_test.py`: separate Unreal-NH training experiment. It selects checkpoints using the test split and should not be treated as a held-out evaluation protocol.

The source code uses the manuscript names **SFDA** and **DSRM**. The renaming changes class and configuration identifiers only; it does not change the module operations or checkpoint parameter keys.

## Environment

Use Python 3.8 or newer and install a PyTorch/Torchvision pair compatible with your CUDA version, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Data layout

Datasets are not included. Place the paired synthetic datasets under `dataset/` and keep hazy and clear filenames compatible with `data/data_loader.py`:

```text
dataset/
├── NHR/
│   ├── train/{hazy,clear}/
│   └── test/{hazy,clear}/
├── NHM/
│   ├── train/{hazy,clear}/
│   └── test/{hazy,clear}/
└── UNREAL_NH/
    ├── train/{hazy,clear}/
    └── test/{hazy,clear}/
```

For self-refinement, also provide the RWNHC, NHRW, and REAL-NH images at the locations described by the corresponding `--rwnhc_root`, `--nhrw_root`, and `--real_nh_root` options. Supply the MUSIQ-AVA and NIQE model files through `--musiq_weights` and `--niqe_weights` when they are not available under `weights/`.

## Training

Run supervised training from the repository root. Use the same epoch count, iterations per epoch, dataset split, and checkpoint as the manuscript experiment; those run-specific values must be filled in from the original experiment record.

```bash
python train_1.py \
  --dataset NHR \
  --dataset_root dataset \
  --epochs <EPOCHS> \
  --iters_per_epoch <ITERATIONS_PER_EPOCH> \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --bs 4 \
  --w_loss_L1 1.0 \
  --w_loss_FFT 0.0 \
  --fusion_variant concat \
  --sfda_variant spatial_frequency_full \
  --dsrm_variant dsrm_full
```

Training outputs are written under `outputs/experiment/`. The training script currently evaluates the configured `test` directory during training and uses its PSNR to select checkpoints. For a strictly held-out test result, use a separate validation split for checkpoint selection and reserve the test split for the final evaluation.

## Self-refinement

After supervised training, run the self-refinement script with the matching checkpoint and dataset roots:

```bash
python self_refinement_nhm.py \
  --nhm_checkpoint <PATH_TO_NHM_CHECKPOINT> \
  --nhm_root dataset/NHM \
  --rwnhc_root dataset/RWNHC_MM23 \
  --nhrw_root dataset/NHRW/NHRW \
  --real_nh_root dataset/REAL-NH_50
```

The default run settings are defined in `self_refinement_nhm.py` and include seed 666, 3,000 maximum updates, evaluation every 250 updates, and patience 3. The script writes checkpoints and metrics under `outputs/self-refinement/` by default.

## Licensing and third-party code

This repository is released under the MIT License. The SFDA/DSRM implementation includes code adapted with authorization from [DEA-Net](https://github.com/cecret3350/DEA-Net). Third-party datasets and IQA weights remain subject to their own terms and are not included here.
