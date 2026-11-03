# Dropping Tokens Without Dropping Them: Causal Attribution for Diabetic Retinopathy

**Pramath Girish Haritz, Chandra Sekhar Seelamantula** · Indian Institute of Science · submitted to ISBI 2027

This repository has the code and model weights of the paper. A vision transformer with a multi-head attention-pooling
(MAP) head grades diabetic retinopathy (DR). The head explains each grade in closed form, with three maps per image:

| Map | Formula | What it tells you | Paper |
|---|---|---|---|
| **Attention weights** ā_t | Σ_h a_{h,t} / H | where the head looks | Sec. 2 |
| **Contribution** c_t | Σ_h a_{h,t} e_{h,t} | how much each patch adds to the grade score S (sums exactly to S − S_∅) | Eq. (2) |
| **Pull** p_t | Σ_h a_{h,t} / (1 − a_{h,t}) · (e_{h,t} − E_h) | how S changes when the patch is removed and the attention re-spreads | Eq. (3) |

The contribution and the pull are linked by the identity **c_{h,t} = (1 − a_{h,t}) p_{h,t} + a_{h,t} E_h** (Eq. 4).

The repository trains the grader, computes the maps and four baselines, and runs every causal test and
lesion-localisation experiment of the paper.

## Contents

- [Dropping Tokens Without Dropping Them: Causal Attribution for Diabetic Retinopathy](#dropping-tokens-without-dropping-them-causal-attribution-for-diabetic-retinopathy)
  - [Contents](#contents)
  - [Quick start](#quick-start)
  - [Three ways to use this repository](#three-ways-to-use-this-repository)
    - [1. Reproduce the paper's tables and figures](#1-reproduce-the-papers-tables-and-figures)
    - [2. Rerun the experiments from the released weights (GPU)](#2-rerun-the-experiments-from-the-released-weights-gpu)
    - [3. Retrain the models](#3-retrain-the-models)
  - [Where each result of the paper comes from](#where-each-result-of-the-paper-comes-from)
  - [Data](#data)
  - [Model weights](#model-weights)
  - [Compute](#compute)
  - [Repository layout](#repository-layout)
  - [Implementation notes](#implementation-notes)
  - [License and data terms](#license-and-data-terms)
  - [Acknowledgements](#acknowledgements)

## Quick start

Install [uv](https://docs.astral.sh/uv/). On first use it creates the pinned environment (Python 3.14, PyTorch 2.11
with CUDA 12.8, timm 1.0.30).

```bash
git clone https://github.com/pramath-haritz/dr-xai.git && cd dr-xai
uv sync
uv run python causal_tests.py report --from paper_results    # prints Table 1 (N, Su, FC); no GPU, no data
```

## Three ways to use this repository

| | What you get | Needs |
|---|---|---|
| **1. Reproduce** | the paper's tables and figures, to the last digit | nothing (the grading table needs the label files of the four grading datasets; Fig. 1 needs the Messidor image) |
| **2. Rerun** | the experiments, from the released weights | a GPU, MAPLES-DR + Messidor (+ DDR) |
| **3. Retrain** | your own models | a GPU, EyePACS, APTOS 2019, BRSET, Messidor-2 |

### 1. Reproduce the paper's tables and figures

`paper_results/` holds the per-image outputs behind every table and figure. Each experiment script has a `report`
step that turns them into the paper's tables, with the same bootstrap draws, so it prints the paper's numbers exactly.

```bash
uv run python evaluate.py report --from paper_results                 # grading: QWK, AUC (needs the label files, see Data)
uv run python causal_tests.py report --from paper_results             # Table 1: N, Su, FC
uv run python lesion_ap.py report --from paper_results                # Table 1: lesion AP
uv run python causal_tests.py report --from paper_results --head ce   # Table 2
uv run python lesion_ap.py report --from paper_results --head ce      # Table 2: AP
uv run python pool_identity.py report --from paper_results            # Fig. 1, Fig. 2 and their numbers
```

> [!NOTE]
> Fig. 1 draws its fundus image from the original Messidor in `data/messidor/`, because the Messidor terms do not
> allow the image to be shared (see [License and data terms](#license-and-data-terms)). Its lesion outlines are
> already in `paper_results/`. Without Messidor, the report skips Fig. 1 and prints everything else.

### 2. Rerun the experiments from the released weights (GPU)

```bash
uv run python download_weights.py            # six checkpoints, 2.2 GB (CORAL and cross-entropy, seeds 0-2)
# put MAPLES-DR, the original Messidor and DDR in data/ (see Data)
uv run python prepare_data.py --check        # says which steps your data allows

uv run python pool_identity.py run           # Fig. 1, Fig. 2          ~5 min per seed
uv run python causal_tests.py run            # Table 1: N, Su, FC      ~6 h per seed, plus the occlusion maps
uv run python lesion_ap.py run               # Table 1: lesion AP      reuses the maps; DDR occlusion ~9 h per seed
uv run python causal_tests.py run --head ce  # Table 2
uv run python lesion_ap.py run --head ce     # Table 2: AP
```

Then run the `report` commands of step 1 without `--from`; they read `outputs/`.

- Every `run` takes `--seeds 0 1 2` (default: all three).
- `--limit N` runs a smoke test on the first N images and writes it to `outputs/smoke/`.
- The grade thresholds come with the checkpoints, so these experiments need no grading data.

### 3. Retrain the models

```bash
# put EyePACS, APTOS 2019, BRSET and Messidor-2 in data/ (see Data)
uv run python prepare_data.py                # checks the data and caches the fundus crops (~15 GB)
for s in 0 1 2; do
  uv run python train.py --seed $s           # CORAL head: stage 1 (1,024 px) + stage 2 (1,536 px), then evaluate
  uv run python train.py --seed $s --head ce # cross-entropy head (Table 2)
done
uv run python evaluate.py report             # grading of your models
```

`train.py` writes `checkpoints/<head>_seed<s>.safetensors`, replacing a downloaded checkpoint of the same name, and
fits its grade thresholds. Steps 2 and 1 then run on your models. `evaluate.py run` re-evaluates any checkpoint.

## Where each result of the paper comes from

Each script has a `run` step (computes, writes per-image outputs) and a `report` step (prints the table or draws the
figure).

| Paper | What | Script | Output of `run` |
|---|---|---|---|
| Sec. 3, *Model and training*: QWK 0.836, AUC 0.972 (Messidor-2) | grading | `train.py`, `evaluate.py` | `checkpoints/`, `outputs/evaluate/` |
| Fig. 1 | the identity on one image | `pool_identity.py` | `outputs/pool_identity/fig1_example.json`, `fig1_lesions.png` |
| Fig. 2; *Closed-form causality*: r = 1.000, 0.996, 0.854, 0.30 vs 0.96 | removal from the pool | `pool_identity.py` | `outputs/pool_identity/coral_seed<s>.json` |
| Table 1, causal tests; *Grade-level causal tests* | N, Su, FC, Holm tests | `causal_tests.py` | `outputs/causal_tests/coral_seed<s>.json` |
| Table 1, lesion AP; *Lesions without pixel labels* | AP, AUC (MAPLES-DR, DDR) | `lesion_ap.py` | `outputs/lesion_ap/coral_seed<s>.json` |
| Table 1, cost per image | TFLOPs (times depend on the GPU) | `causal_tests.py profile` | printed |
| Table 2; *The ordinal head shapes the attention* | CORAL vs cross-entropy | `causal_tests.py --head ce`, `lesion_ap.py --head ce`, `evaluate.py` | `outputs/*/ce_seed<s>.*` |

The HRDecoder and nnU-Net rows of Table 1 are supervised segmentation models that are not part of this repository;
`lesion_ap.py report` quotes their APs from the paper.

## Data

- **No dataset can be redistributed here**, so download each one from its source.
- Put each dataset in its own folder under `data/` and unpack it there. The code finds images by file name, so the
  layout inside each folder does not matter.
- To keep the data elsewhere, pass `--data-root /path` or set `DR_XAI_DATA=/path`.
- The train / val / test splits are in `splits/grading.csv` (columns: key, dataset, image file name, split). **The
  grades are not redistributed:** the code reads them from each dataset's own label files (second column of the
  table below), so keep those files in the dataset's folder. `prepare_data.py --check` reports whether they are found.

| Folder | Dataset (what to put there) | Used for | Source and access |
|---|---|---|---|
| `data/eyepacs/` | EyePACS (Kaggle *Diabetic Retinopathy Detection*): `train/`, `test/`; labels `trainLabels.csv`, `retinopathy_solution.csv` (unzipped) | training, val, test | [Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data): accept the competition rules; 88 GB |
| `data/aptos2019/` | APTOS 2019 Blindness Detection: `train_images/`; labels `train.csv` | training, val | [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data): accept the competition rules |
| `data/brset/` | BRSET: `fundus_photos/`; labels `labels_brset.csv` (`labels.csv` in older versions) | training, val, test | [PhysioNet](https://physionet.org/content/brazilian-ophthalmological/): **credentialed access** (training and a data use agreement) |
| `data/messidor2/` | Messidor-2: all 1,748 images; labels `messidor_data.csv` (Krause et al. grades) | out-of-distribution grading test | [ADCIS](https://www.adcis.net/en/third-party/messidor2/): licence form. Grades of [Krause et al.](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades), CC0 |
| `data/messidor/` | Messidor, the original: 1,200 images, `.tif` or `.png` | the images of MAPLES-DR | [ADCIS](https://www.adcis.net/en/third-party/messidor/): licence form |
| `data/maples-dr/` | MAPLES-DR v3, `MAPLES-DR.zip` unpacked: `train/`, `test/` | causal tests, lesion AP, figures | [figshare](https://doi.org/10.6084/m9.figshare.24328660) |
| `data/ddr/` | DDR, `lesion_segmentation/test/` (`image/`, `label/{EX,HE,MA,SE}/`) | lesion AP | [GitHub](https://github.com/nkicsl/DDR-dataset): Google Drive or Baidu links |

**What each step needs**

| Step | Datasets |
|---|---|
| reports from `paper_results/` | none; `evaluate.py report`: the label files of EyePACS, APTOS 2019, BRSET and Messidor-2 (no images); Fig. 1: Messidor, for its image |
| `pool_identity.py`, `causal_tests.py` | MAPLES-DR + Messidor |
| `lesion_ap.py` | MAPLES-DR + Messidor, DDR |
| `evaluate.py` | the val and test images of EyePACS, APTOS 2019 and BRSET; Messidor-2 |
| `train.py` | EyePACS, APTOS 2019, BRSET (train and val) |

**Checking the data.** `uv run python prepare_data.py --check` counts, for every dataset, how many of the images in
`splits/*.csv` it finds, and lists the steps the present data allows. Without `--check` it also writes the fundus
crops of the grading images to `cache/crops/` (the square around the fundus disk, at most 1,536 px; about 15 GB).

**How the splits were made.** They were frozen before training; the code that built them is not needed and not
included.

- EyePACS: images rated *Good* or *Usable* by [EyeQ](https://github.com/HzFu/EyeQ), split by patient 90 / 5 / 5.
- BRSET: images of adequate quality, split by patient.
- APTOS 2019: de-duplicated; train and val only.
- Any training image within 8 bits (256-bit perceptual hash) of an evaluation or lesion-set image was removed.

This gives **34,032 training, 3,813 validation and 3,267 test images**, and the **1,744 gradable Messidor-2 images**.

## Model weights

Six checkpoints (safetensors, 372 MB each) are on the [GitHub release](https://github.com/pramath-haritz/dr-xai/releases).
`weights.json` lists them with their SHA-256, and `download_weights.py` fetches and verifies them.

| File | Head | Seeds |
|---|---|---|
| `coral_seed{0,1,2}.safetensors` | CORAL (ordinal), the model of the paper | 0, 1, 2 |
| `ce_seed{0,1,2}.safetensors` | 5-class cross-entropy, the comparison of Table 2 | 0, 1, 2 |

The metadata of each file holds the head, the seed, the training configuration of both stages, and the four grade
thresholds and the validation SD of the grade score that `evaluate.py` fitted.

**Using a checkpoint in your own code:**

```python
import numpy as np
from PIL import Image
from drxai import attribution, data
from drxai.model import GRID, grade, input_tensor, load

model, meta = load("checkpoints/coral_seed0.safetensors")              # meta["thresholds"]: T_1..T_4 on the grade score
crop = np.asarray(data.fundus_crop(Image.open("fundus.jpg").convert("RGB")))   # square crop around the fundus disk
d = attribution.decompose(model, input_tensor(crop))                   # 1,536-px input: 96 x 96 patches
maps = attribution.maps(d, GRID)                                       # attention, contribution, pull, attention_term
print(grade(d["S"], meta["thresholds"]), d["S_empty"] + d["c"].sum().item() - d["S"])   # grade; completeness (~0)
```

## Compute

Times are on one NVIDIA RTX PRO 6000 with the paper's batch sizes. Peak GPU memory is as allocated by PyTorch (each
`run` prints its own), so **a 24 GB GPU runs everything, training included.**

| Step | Time per seed | Peak GPU memory |
|---|---|---|
| `train.py` (one head) | 2 h (stage 1, 1,024 px) + 1.3 h (stage 2, 1,536 px) | 6.3 GB, 13.7 GB (batch 32, gradient checkpointing) |
| `evaluate.py run` | ~10 min per checkpoint | 2.1 GB |
| `pool_identity.py run` | ~5 min (Fig. 1 adds under a minute on seed 1) | 0.8 GB |
| maps of the head (attention weights, contribution, pull, attention term) | seconds | 0.6 GB |
| rollout, Grad-CAM, AttnLRP | ~5 min (MAPLES-DR) | 2.3, 4.5, 8.7 GB |
| occlusion | ~5 h (MAPLES-DR), ~9 h (DDR) | ≤ 8.7 GB (`--occlusion-batch 32`) |
| `causal_tests.py run` | ~6 h (CORAL, 7 maps), ~45 min (CE) | 4.5 GB (`--batch 24`) |

- `causal_tests.py profile` prints the cost columns of Table 1 for your GPU. It reproduces the paper's TFLOPs: 4.7 for
  the forward pass and our maps, 6.3 rollout, 14.2 Grad-CAM, 25.3 AttnLRP, and about 8,400 occlusion, depending on the
  image.
- The maps are cached in `outputs/maps/` and shared between `causal_tests.py` and `lesion_ap.py`.
- On a smaller GPU, lower `--batch` (token dropping) and `--occlusion-batch`; the results change only at the level of
  bfloat16 rounding.
- Training keeps batch 32 in one step, because the spread term of the loss (see
  [Implementation notes](#implementation-notes)) acts on the whole batch.

## Repository layout

```
prepare_data.py      check data/ against the splits; cache the fundus crops
download_weights.py  fetch and verify the released checkpoints
train.py             train the grader (CORAL or --head ce), two stages, then evaluate
evaluate.py          run: grade thresholds and predictions | report: grading table
pool_identity.py     run: completeness, identity, removal from the pool | report: Fig. 1, Fig. 2
causal_tests.py      run: token-dropping tests | report: Table 1 (causal), Table 2 | profile: cost per image
lesion_ap.py         run: lesion AP of every map | report: Table 1 (AP), Table 2 (AP)
drxai/
  data.py            splits, fundus crops, MAPLES-DR and DDR images and masks
  model.py           NaFlexViT-B/16 + MAP head + CORAL / cross-entropy head; checkpoints
  attribution.py     attention weights, contribution, pull, head evidence, alignment q; removal from the pool (Eqs. 1-4)
  baselines.py       attention rollout, Grad-CAM, AttnLRP, occlusion
  maps.py            computes and caches all maps of a model on a lesion test set
  removal.py         token dropping; necessity, sufficiency, faithfulness correlation
  stats.py           hierarchical bootstrap, paired comparisons, Holm correction
  figures.py         Fig. 1 and Fig. 2
splits/              grading.csv (42,856 images), lesion.csv (MAPLES-DR 198, DDR 225)
paper_results/       the per-image outputs behind the paper's tables and figures
weights.json         release URL and SHA-256 of the checkpoints
data/ cache/ checkpoints/ outputs/   (not tracked) datasets, crops, weights, your results
```

## Implementation notes

**Grade score and thresholds**
- S is the CORAL score (logit_k = S + b_k) or, for the cross-entropy head, the expected grade.
- Training samples grade 0 and grade > 0 equally, so the four thresholds are refitted on the validation split: a
  logistic regression of [grade ≥ k] on each CORAL logit, or on the expected grade.
- The predicted grade is the number of thresholds S exceeds.

**Training loss**
- Besides the CORAL (or cross-entropy) loss, training adds 0.1 × a VICReg-style spread term on a 32-d projection of
  the pooled feature: a hinge on the per-dimension standard deviation plus the mean squared correlation between
  dimensions.
- The spread term acts on the whole batch, so batch 32 is part of the recipe.
- The projection is not used at inference and is not in the released weights.

**Maps**
- The head is evaluated in float64 on the tokens of a bfloat16 backbone pass.
- The read-out g of Eq. (1) is the integrated gradient of φ on 64 midpoints from the empty pool b_O to z.
- Rollout: from the pool query.
- Grad-CAM: on the tokens after block 8.
- AttnLRP: CP-LRP and the gamma rule (γ = 0.1 on linear layers and on the patch embedding), the configuration with
  the best lesion AP of 30 tried on IDRiD.
- Occlusion: 2 × 2-patch windows, filled by a masked Gaussian blur (σ = 16 px).

**Causal tests**
- Patches are removed by dropping their tokens after the patch and position embedding. Tokens outside the field of
  view are always kept.
- Fractions: 0.25% to 50% of the fundus patches. Ties are broken by a fixed random order.
- The random draws of each image (tie order, random orders, the 128 FC groups) come from a generator seeded with
  (2000 + model seed, row of the image in `splits/lesion.csv`). `pool_identity.py` reuses the same FC groups.

**Statistics**
- Hierarchical bootstrap over images and seeds: 5,000 resamples for tests, 2,000 for intervals; two-sided.
- Holm correction over the 12 comparisons of each map (6 for Table 2).
- The reports make their draws from fixed-seed generators in a fixed order, so `report --from paper_results`
  reproduces the paper's intervals and p-values to the last digit.
- A new `run` matches the paper up to bfloat16 rounding, which depends on the GPU and the batch size.

## License and data terms

- **Code and model weights:** [CC BY-NC 4.0](LICENSE): non-commercial use, with attribution.
- **Weights and datasets:** the weights were trained on EyePACS, APTOS 2019 and BRSET; their use is also subject to
  the terms of those datasets.
- **Fig. 1 lesion outlines:** `paper_results/pool_identity/fig1_lesions.png` is derived from the expert masks of
  MAPLES-DR (CC BY 4.0).
- **Fig. 1 image:** from the Messidor database, kindly provided by the Messidor program partners (see
  https://www.adcis.net/en/third-party/messidor/). Its terms do not allow redistribution, so it is not included.

> [!WARNING]
> The models are for research purposes only, not medical devices. Do not use them for clinical decisions.

## Acknowledgements

We gratefully acknowledge funding from ZEISS India and the Kotak-IISc AI/ML Centre.
