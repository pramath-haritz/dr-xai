"""Download the released checkpoints listed in weights.json into checkpoints/ and verify their SHA-256.

    uv run python download_weights.py                  # all six: CORAL and cross-entropy heads, seeds 0-2 (2.2 GB)
    uv run python download_weights.py --head coral     # the three CORAL checkpoints (Table 1, Fig. 1, Fig. 2)

Each checkpoint (safetensors, 372 MB) holds the weights of the grader and, in its metadata, the head, the seed, the
training configuration, and the grade thresholds and validation SD of the grade score fitted by evaluate.py.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", choices=["coral", "ce"], default=None, help="only the checkpoints of this head")
    args = ap.parse_args()
    spec = json.load(open(REPO / "weights.json"))
    if "<" in spec["url"]:
        raise SystemExit(f"weights.json has no release URL yet ({spec['url']})")
    out = REPO / "checkpoints"
    out.mkdir(exist_ok=True)
    for name, info in spec["files"].items():
        if args.head and not name.startswith(args.head + "_"):
            continue
        path = out / name
        if path.exists() and sha256(path) == info["sha256"]:
            print(f"{name}: present and verified")
            continue
        print(f"{name}: downloading {info['bytes'] / 1e6:.0f} MB ...", flush=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"] + name, part)
        if sha256(part) != info["sha256"]:
            part.unlink()
            raise SystemExit(f"{name}: SHA-256 mismatch, the download is corrupt; please try again")
        part.rename(path)
        print(f"{name}: verified")


if __name__ == "__main__":
    main()
