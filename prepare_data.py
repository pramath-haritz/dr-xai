"""Check the datasets in data/ against the frozen splits, then cache the fundus crops of the grading images.

    uv run python prepare_data.py               # check, then crop every grading image that is present
    uv run python prepare_data.py --check       # check only

The check reports, for each dataset, how many of the images in splits/*.csv are present, whether its label files
(the source of the grades, which are not redistributed) are present and grade every image, and which steps of the
pipeline the present data allow. Crops: cache/crops/<key>.jpg, the square around the fundus disk at full resolution,
downsampled to 1,536 px if larger (JPEG quality 95), about 15 GB for all 42,856 grading images. The lesion test sets
(MAPLES-DR, DDR) need no preparation: the experiments read them from data/.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from drxai import data

GRADING = {"eyepacs": "EyePACS", "aptos2019": "APTOS 2019", "brset": "BRSET", "messidor2": "Messidor-2"}
LESION = {"maples": "MAPLES-DR (masks: data/maples-dr, images: original Messidor in data/messidor)",
          "ddr": "DDR (lesion segmentation test set: data/ddr)"}


def check():
    rows = data.grading_rows(with_grades=False)
    present, graded = {}, {}
    print(f"data root: {data.data_root()}\n\nGrading images (splits/grading.csv):")
    for ds, label in GRADING.items():
        rs = [r for r in rows if r["dataset"] == ds]
        found = [r for r in rs if data.find(ds, r["image"]) is not None]
        present[ds] = len(found) == len(rs)
        print(f"  {label:12s} {len(found):6d} / {len(rs):6d} images" + ("" if present[ds] else
              f"   missing, e.g. {next(r['image'] for r in rs if data.find(ds, r['image']) is None)} in data/{ds}/"))
        g = data.grades(ds)
        if g is None:
            need = " and ".join(" or ".join(n) for p, n in data.label_files(ds) if p is None)
            print(f"  {'':12s} labels missing: {need} in data/{ds}/")
        else:
            n = sum(Path(r["image"]).stem.lower() in g for r in rs)
            print(f"  {'':12s} {n:6d} / {len(rs):6d} grades from " + ", ".join(p.name for p, _ in data.label_files(ds)))
        graded[ds] = g is not None and n == len(rs)
    print("\nLesion test sets (splits/lesion.csv):")
    for ds, label in LESION.items():
        rs = data.lesion_rows(ds)
        ok = 0
        for name, sub in rs:
            img, masks = data.lesion_files(ds, name, sub)
            ok += img is not None and all(m is not None for m in masks.values())
        present[ds] = ok == len(rs)
        print(f"  {label}: {ok} / {len(rs)} images with all four masks")
    grading = all(present[d] and graded[d] for d in GRADING)
    print("\nWith this data you can run:")
    print(f"  train.py, evaluate.py          {'yes' if grading else 'no (needs all four grading datasets and their labels)'}")
    print(f"  evaluate.py report             {'yes' if all(graded.values()) else 'no (needs the labels of the four grading datasets)'}")
    for script, need in (("pool_identity.py", ("maples",)), ("causal_tests.py", ("maples",)), ("lesion_ap.py", ("maples", "ddr"))):
        print(f"  {script:30s} {'yes' if all(present[d] for d in need) else 'no (needs ' + ', '.join(LESION[d].split(' (')[0] for d in need) + ')'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check the data, write nothing")
    ap.add_argument("--data-root", default=None, help="folder with the dataset folders (default: data/)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    data.set_data_root(args.data_root)
    rows = check()
    if args.check:
        return
    todo = [(r["key"], data.find(r["dataset"], r["image"])) for r in rows if not (data.CROPS / f"{r['key']}.jpg").exists()]
    todo = [(k, p) for k, p in todo if p is not None]
    if not todo:
        print(f"\nAll present grading images are cropped ({data.CROPS}).")
        return
    data.CROPS.mkdir(parents=True, exist_ok=True)
    print(f"\nCropping {len(todo)} grading images to {data.CROPS} ...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(data.write_crop, *zip(*todo), chunksize=16)):
            if (i + 1) % 2000 == 0 or i + 1 == len(todo):
                print(f"  {i + 1} / {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
