"""Data: the frozen splits, the fundus crops of the grading images and the lesion test sets (MAPLES-DR, DDR).

Raw datasets live in data/<dataset>/ (or --data-root / DR_XAI_DATA), each unpacked as downloaded. Images are found by
file name anywhere inside their dataset folder; the masks of MAPLES-DR and DDR by the folder structure of their release
(train|test/<lesion>/ and lesion_segmentation/test/label/<type>/), which may sit at any depth.

  grading  splits/grading.csv: EyePACS, APTOS 2019, BRSET (train / val / test) and Messidor-2 (out-of-distribution test).
           The grades come from each dataset's own label files (LABEL_FILES). The model reads square fundus crops,
           cached once by prepare_data.py in cache/crops/<key>.jpg.
  lesion   splits/lesion.csv: MAPLES-DR (198 images of the original Messidor; expert masks of MA, HE, EX and SE) and
           the DDR lesion-segmentation test set (225 images). Read straight from data/; the row order of MAPLES-DR
           seeds the per-image random generators of the causal tests.
"""
import csv
import functools
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache"
CROPS = CACHE / "crops"
CAP = 1536                    # crops wider than this are downsampled to it (the largest input side of the model)
GRID = 1024                   # MAPLES-DR masks for the lesion control of the causal tests are max-pooled to this grid
SCORE_SIDE = 1536             # lesion AP is scored on 1,536 x 1,536 crops
LESIONS = ("MA", "HE", "EX", "SE")
MAPLES_FOLDERS = {"MA": "Microaneurysms", "HE": "Hemorrhages", "EX": "Exudates", "SE": "CottonWoolSpots"}
IMAGE_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def data_root():
    return Path(os.environ.get("DR_XAI_DATA", REPO / "data"))


def set_data_root(path):
    if path:
        os.environ["DR_XAI_DATA"] = str(Path(path).resolve())
        index.cache_clear()
        find_dir.cache_clear()
        grades.cache_clear()


# ----------------------------------------------------------------------------- splits
# The grades are not redistributed: they are read from each dataset's own label files, found by file name in
# data/<dataset>/. Per dataset: [(accepted file names, image column, grade column)], every entry is needed.
LABEL_FILES = {
    "eyepacs": [(("trainLabels.csv", "train.csv"), "image", "level"),
                (("retinopathy_solution.csv", "test.csv"), "image", "level")],
    "aptos2019": [(("train.csv",), "id_code", "diagnosis")],
    "brset": [(("labels_brset.csv", "labels.csv"), "image_id", "DR_ICDR")],
    "messidor2": [(("messidor_data.csv",), "image_id", "adjudicated_dr_grade")],
}


def label_files(dataset):
    """[(path or None, accepted file names)] of the label files of a grading dataset"""
    return [(next((p for n in names if (p := find(dataset, n)) is not None), None), names)
            for names, _, _ in LABEL_FILES[dataset]]


@functools.cache
def grades(dataset):
    """{lower-case image stem: grade} from the dataset's own label files (None if one of them is missing)"""
    out = {}
    for (path, _), (_, img_col, grade_col) in zip(label_files(dataset), LABEL_FILES[dataset]):
        if path is None:
            return None
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            if r[grade_col].strip():
                out[Path(r[img_col]).stem.lower()] = int(float(r[grade_col]))
    return out


def grading_rows(with_grades=True):
    """splits/grading.csv as a list of dicts, in the frozen row order; with_grades: each row's grade (int) from the
    label files of its dataset"""
    rows = list(csv.DictReader(open(REPO / "splits" / "grading.csv")))
    if not with_grades:
        return rows
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        g = grades(ds)
        if g is None:
            need = " and ".join(" or ".join(n) for p, n in label_files(ds) if p is None)
            raise SystemExit(f"the grades of {ds} are read from its own label files: put {need} in "
                             f"{data_root() / ds}/ (see README, Data)")
        rs = [r for r in rows if r["dataset"] == ds]
        missing = [r["image"] for r in rs if Path(r["image"]).stem.lower() not in g]
        if missing:
            raise SystemExit(f"{len(missing)} {ds} images of splits/grading.csv have no grade in its label files "
                             f"(e.g. {missing[0]}): check that they are the files of the release named in the README")
        for r in rs:
            r["grade"] = g[Path(r["image"]).stem.lower()]
    return rows


def lesion_rows(dataset):
    """[(name, subset)] of one lesion test set ('maples' or 'ddr'), in the frozen row order"""
    return [(r["name"], r["subset"]) for r in csv.DictReader(open(REPO / "splits" / "lesion.csv")) if r["dataset"] == dataset]


@functools.cache
def index(dataset):
    """{lower-case file name: path} of every file under data/<dataset>/"""
    out = {}
    for d, _, files in os.walk(data_root() / dataset, followlinks=True):
        for f in files:
            out.setdefault(f.lower(), Path(d) / f)
    return out


def find(dataset, filename):
    """path of a file of a dataset by its file name (None if missing)"""
    return index(dataset).get(filename.lower())


def find_stem(dataset, stem):
    """path of an image by its file stem (any of IMAGE_EXT)"""
    for ext in IMAGE_EXT:
        p = find(dataset, stem + ext)
        if p is not None:
            return p
    return None


@functools.cache
def find_dir(dataset, marker):
    """the folder D inside data/<dataset>/ for which D/marker exists (None if there is none)"""
    for d, _, _ in os.walk(data_root() / dataset, followlinks=True):
        if (Path(d) / marker).is_dir():
            return Path(d)
    return None


# ----------------------------------------------------------------------------- fundus crops
def fundus_box(img):
    """(left, top, right, bottom) of the square around the fundus disk: the horizontal extent of the pixels brighter
    than 20 (on a 512-px-wide copy) gives the diameter, its centre and the vertical centre give the centre"""
    w, h = img.size
    s = 512 / w
    small = np.asarray(img.resize((512, max(1, round(h * s))), Image.BILINEAR)).max(axis=2)
    mask = small > 20
    if mask.sum() < 1000:                                         # blank frame: keep the whole image
        cx, cy, d = w / 2, h / 2, max(w, h)
    else:
        cols, rows = np.where(mask.any(0))[0], np.where(mask.any(1))[0]
        x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        cx, cy, d = (x0 + x1) / 2 / s, (y0 + y1) / 2 / s, (x1 - x0) / s
    r = d / 2
    return (round(cx - r), round(cy - r), round(cx + r), round(cy + r))


def fundus_box_relative(img):
    """fundus_box with the threshold relative to the image's own background (some DDR images have a grey border);
    used for the 1,536-px crops on which lesion AP is scored"""
    w, h = img.size
    s = 512 / w
    small = np.asarray(img.resize((512, max(1, round(h * s))), Image.BILINEAR)).max(axis=2).astype(int)
    background = np.median(np.concatenate([small[:, :3].ravel(), small[:, -3:].ravel()]))
    mask = small > max(20, background + 20)
    if mask.sum() < 1000:
        cx, cy, d = w / 2, h / 2, max(w, h)
    else:
        cols, rows = np.where(mask.any(0))[0], np.where(mask.any(1))[0]
        x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        cx, cy, d = (x0 + x1) / 2 / s, (y0 + y1) / 2 / s, (x1 - x0) / s
    r = d / 2
    return (round(cx - r), round(cy - r), round(cx + r), round(cy + r))


def fundus_crop(img):
    """square fundus crop at full resolution, downsampled (Lanczos) only if wider than CAP; crop() pads with black"""
    crop = img.crop(fundus_box(img))
    if crop.size[0] > CAP:
        crop = crop.resize((CAP, CAP), Image.LANCZOS)
    return crop


def write_crop(key, path):
    """cache/crops/<key>.jpg (JPEG quality 95) of one grading image; returns the key"""
    dst = CROPS / f"{key}.jpg"
    if not dst.exists():
        fundus_crop(Image.open(path).convert("RGB")).save(dst, quality=95)
    return key


def load_crop(key, side):
    """cached crop resized to side x side (the input of the grading model)"""
    img = Image.open(CROPS / f"{key}.jpg")
    img.draft("RGB", (side, side))                                # JPEG: decode at a reduced scale if >= 2x side
    return img.convert("RGB").resize((side, side), Image.BILINEAR, reducing_gap=2.0)


class Crops(torch.utils.data.Dataset):
    """uint8 CHW tensors of the cached crops (augmentation happens on the GPU), with their position in keys"""

    def __init__(self, keys, side):
        self.keys, self.side = keys, side

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(load_crop(self.keys[i], self.side))).permute(2, 0, 1), i


def crop_loader(keys, side, batch, workers=16):
    """batches in the order of keys"""
    return torch.utils.data.DataLoader(Crops(keys, side), batch_size=batch, shuffle=False, num_workers=workers,
                                       pin_memory=True)


# ----------------------------------------------------------------------------- lesion test sets
def lesion_files(dataset, name, subset):
    """(image path, {lesion type: mask path or None}); None where the file is not in data/ (a missing mask counts as
    a mask without lesions)"""
    exists = lambda p: p if p is not None and p.exists() else None
    if dataset == "maples":                       # the expert masks of MAPLES-DR; its images are the original Messidor's
        root = find_dir("maples-dr", "train/Microaneurysms")
        masks = {t: exists(root / subset / MAPLES_FOLDERS[t] / f"{name}.png") if root else None for t in LESIONS}
        return find_stem("messidor", name), masks
    root = find_dir("ddr", "test/label/MA")                   # DDR: lesion_segmentation/test/{image, label/<type>}
    if root is None:
        return None, {t: None for t in LESIONS}
    return exists(root / "test" / "image" / f"{name}.jpg"), {t: exists(root / "test" / "label" / t / f"{name}.tif")
                                                             for t in LESIONS}


def _grid_mask(path, box, fundus):
    """binary mask on the GRID x GRID grid of the crop; max-pooling (BOX resize, then > 0) keeps every lesion"""
    if path is None:
        return np.zeros((GRID, GRID), bool)
    m = Image.open(path).convert("L").point(lambda v: 255 if v > 0 else 0).crop(box)
    return (np.asarray(m.resize((GRID, GRID), Image.BOX)) > 0) & fundus


def _score_mask(path, size, box):
    """mask in the frame of the 1,536-px scoring crop: area-averaged resize, lesion where >= 0.25"""
    if path is None:
        return np.zeros((SCORE_SIDE, SCORE_SIDE), bool)
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].max(2)
    m = Image.fromarray((a > 0).astype(np.float32))
    assert m.size == size, (path, m.size, size)
    return np.asarray(m.crop(box).resize((SCORE_SIDE, SCORE_SIDE), Image.BOX)) >= 0.25


def load_lesion_image(dataset, name, subset):
    """model input of one lesion test image:
      image    uint8 [h, w, 3] crop given to the model (resized to the model side by model.input_tensor)
      fundus   bool mask that defines the fundus patches (MAPLES-DR: crop resized to 1,024 px, max channel > 20;
               DDR: the 1,536-px scoring crop, max channel > 15)
      lesions  MAPLES-DR only: union of the expert masks on the 1,024 grid (the lesion control of the causal tests)"""
    path, masks = lesion_files(dataset, name, subset)
    if path is None:
        raise SystemExit(f"{dataset} image {name} is not in {data_root()}: see the Data section of README.md")
    img = Image.open(path).convert("RGB")
    if dataset == "maples":
        box = fundus_box(img)
        crop = fundus_crop(img)
        fundus = np.asarray(crop.resize((GRID, GRID), Image.BILINEAR)).max(2) > 20
        lesions = np.any([_grid_mask(masks[t], box, fundus) for t in LESIONS], 0)
        return dict(name=name, image=np.asarray(crop), fundus=fundus, lesions=lesions)
    crop = np.asarray(img.crop(fundus_box_relative(img)).resize((SCORE_SIDE, SCORE_SIDE), Image.BICUBIC))
    return dict(name=name, image=crop, fundus=crop.max(-1) > 15, lesions=None)


def score_crop(path):
    """the 1,536-px scoring crop of an image (bicubic crop with fundus_box_relative), its box and the image size"""
    img = Image.open(path).convert("RGB")
    box = fundus_box_relative(img)
    return np.asarray(img.crop(box).resize((SCORE_SIDE, SCORE_SIDE), Image.BICUBIC)), box, img.size


def load_lesion_targets(dataset, name, subset):
    """expert masks of one lesion test image in the frame of its 1,536-px scoring crop (bicubic crop of the image with
    fundus_box_relative): fm = fundus (max channel > 15), masks of MA, HE, EX, SE and ANY (their union), inside fm,
    and the 8-connected components of each mask"""
    path, paths = lesion_files(dataset, name, subset)
    if path is None:
        raise SystemExit(f"{dataset} image {name} is not in {data_root()}: see the Data section of README.md")
    crop, box, size = score_crop(path)
    fm = crop.max(-1) > 15
    masks = {t: _score_mask(paths[t], size, box) & fm for t in LESIONS}
    masks["ANY"] = np.any([masks[t] for t in LESIONS], 0)
    labels = {t: ndimage.label(m, structure=np.ones((3, 3))) for t, m in masks.items()}
    return dict(name=name, image=crop, fm=fm, masks=masks, labels=labels)
