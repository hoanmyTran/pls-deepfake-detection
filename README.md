<div align="center">

# Progressive Layer-wise Supervision (PLS)

### Deep-to-Shallow Supervision Annealing for Efficient and Robust Speech Deepfake Detection
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202025-Main%20Conference-blue.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/150091)
[![PyTorch](https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Training-Lightning-792EE5)](https://lightning.ai/)

*This repository contains the code for the following NeurIPS 2026 paper*

</div>

## Overview

Standard final-layer fine-tuning can leave shallow SSL representations weakly discriminative. **PLS** attaches the **same shared linear head** to every XLS-R hidden state and anneals deep-layer supervision toward shallower layers.

## Method

At epoch \(e\), the exponential weighting schedule is

$$\rho(e)=M e^{-ke},\qquad w_l(e)=\frac{(l+1)^{\rho(e)}}{\sum_{j=0}^{L}(j+1)^{\rho(e)}}.$$

$$\mathcal L(e)=\begin{cases}\mathcal L_L,&e\text{ even}\\\sum_{l=0}^L w_l(e)\mathcal L_l,&e\text{ odd}.\end{cases}$$

`M` and `k` default to `1.5` and `0.05`. Cross-entropy weights are `[0.1, 0.9]` for spoof (0) and bona fide (1).

## Components and data flow

```text
config.yaml ─┐
             ├─> main.py ─> LightningPLS (training orchestration)
protocol.csv ┘                    │
     │                            ├─ PLSClassifier : torch.nn.Module
     v                            │   └─ XLS-R → per-layer mean pooling → shared linear classifier
AudioDataset → DataLoader ────────┤
     │                            └─ ProgressiveLayerwiseLoss : torch.nn.Module
     └─ training only: RawBoost adapter → original RawBoost.py (unchanged)

trainer.fit → val_loss / val_accuracy → best epoch checkpoints (.ckpt)
trainer.test → LayerScoreWriter → score_fp.csv, score_b0.csv ... score_b23.csv
                                           ↓
                                EER per dataset / average / pooled
```

| Component | Responsibility |
|---|---|
| `models/pls_classifier.py` | Pure PyTorch XLS-R and shared classifier; forward can return final or all-layer logits |
| `losses/progressive_loss.py` | Pure PyTorch PLS exponential schedule and even/odd epoch objective |
| `trainers/lightning_module.py` | Optimizer, epoch metrics, training/validation steps, and test writer |
| `data/audio.py` | CSV protocol parsing, audio load, repeat padding or batch padding |
| `data/lightning_data.py` | Lightning `DataModule` for train/validation/test loaders |
| `data/rawboost_adapter.py` | Train-only augmentation; imports original `RawBoost.py` when enabled |
| `evaluation/scorer.py` | `ID,Score,Label` CSVs with `[fake_logit, real_logit]` |
| `evaluate_datasets.py` | Reuses one model/checkpoint for multiple selected datasets |

## Installation

```bash
conda create -n PLS python=3.9
conda activate PLS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Dataset format

```csv
file_name,label
/data/audio/real.wav,bonafide
/data/audio/fake.wav,spoof
```

RawBoost data augmentation example:

```yaml
rawboost:
  algo: 0  # disabled; choose 1-8 for augmentation during training only
```

## Train

```bash
python main.py --config config.yaml
```

## Evaluate one dataset

```bash
python main.py --config config.yaml --eval \
  --checkpoint results/pls_xlsr300m/epoch=...ckpt \
  --protocol_path protocols/21df.csv \
  --output_score_file scores/21df
```

## Evaluate multiple datasets

Edit `datasets.yaml` to set `protocol_path` and `enabled` for each dataset, then run:

```bash
python evaluate_datasets.py --config config.yaml \
  --checkpoint results/pls_xlsr300m/epoch=...ckpt \
  --datasets 21df 25df itw \
  --output-root scores/pls_xlsr300m
```

Or run `CHECKPOINT=... bash run_all_evals.sh`. Each dataset gets its own output folder:

```text
scores/pls_xlsr300m/
├── 21df/score_fp.csv, score_b0.csv, ..., score_b23.csv
├── 25df/score_fp.csv, score_b0.csv, ..., score_b23.csv
├── itw/score_fp.csv, score_b0.csv, ..., score_b23.csv
└── summary.csv
```

Each score file has `ID,Score,Label`; `Score` is the **two original logits**, `[Fake_Logit, Real_Logit]`. For EER, the countermeasure score is `Real_Logit - Fake_Logit`, so larger means more bona fide.

## Compute metrics including pooled and average EERs
```bash
python evaluate_eer_directory.py --score_dir scores/pls/
```

## Framework boundary: plug in and out

```python
from models.pls_classifier import PLSClassifier
from trainers.lightning_module import LightningPLS

network = PLSClassifier('facebook/wav2vec2-xls-r-300m') # plain nn.Module
lightning = LightningPLS(config, model=network)        # plug in for training/testing
network = lightning.model                             # unplug for standalone inference
```

The *model* has no Lightning imports. Standalone inference using the trained network is permitted; there is **no native PyTorch training loop** in this repository. When importing weights, keep track of the distinction between a Lightning `.ckpt` (keys prefixed by `model.`) and exported `model_weights.pt` (plain state dict).


## Acknowledgements
Our implementations use the source code from the following repositories:

- [RawBoost Data Augmentation](https://github.com/TakHemlata/RawBoost-antispoofing)
- [Pretrained XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [Metrics](https://github.com/nii-yamagishilab/AntiDeepfake)

## Citation
```  
@inproceedings{
tran2026progressive,
title={Progressive Layer-wise Supervision: Deep-to-Shallow Supervision Annealing for Efficient and Robust Speech Deepfake Detection},
author={Hoan My Tran and Xin Wang and Xuechen Liu},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=kmOSYc5cfb}
}
```  
