"""Download the Dataset059 training pool: all patches except jeong200 eval patches and any patch touching one
(same scroll, |origin delta| <= 320 on every axis = adjacent cell, incl. diagonals). Writes data/train/ds059/.

  uv run python scripts/fetch_ds059_train.py [--workers 8] [--dry-run]
"""

import argparse
import glob
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = "https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/"
OUT = REPO / "data" / "train" / "ds059"
BUFFER = 320


def parse(s):
    return (s.split("_")[0], *map(int, re.search(r"_z(\d+)_y(\d+)_x(\d+)", s).groups()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    html = urllib.request.urlopen(BASE + "labelsTr/").read().decode()
    allp = sorted({m[:-4] for m in re.findall(r'href="([^"]+\.tif)"', html)})
    ev = {Path(f).stem for f in glob.glob(str(REPO / "data" / "eval" / "jeong200" / "labelsTr" / "*.tif"))}
    E = [parse(s) for s in ev]

    def near(s):
        a = parse(s)
        return any(e[0] == a[0] and all(abs(a[i] - e[i]) <= BUFFER for i in (1, 2, 3)) for e in E)

    train = [s for s in allp if s not in ev and not near(s)]
    print(f"{len(allp)} patches, {len(ev)} eval, {len(train)} train after buffer {BUFFER}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps({"train": train, "excluded_eval": sorted(ev), "buffer": BUFFER,
                                                "source": BASE}, indent=1))
    if args.dry_run:
        return
    jobs = []
    for s in train:
        jobs.append((BASE + f"imagesTr/{s}_0000.tif", OUT / "imagesTr" / f"{s}_0000.tif"))
        jobs.append((BASE + f"labelsTr/{s}.tif", OUT / "labelsTr" / f"{s}.tif"))
    for d in ("imagesTr", "labelsTr"):
        (OUT / d).mkdir(exist_ok=True)

    def get(job):
        url, dst = job
        if dst.exists() and dst.stat().st_size > 0:
            return 0
        tmp = dst.with_suffix(".part")
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(tmp), url])
        if r.returncode == 0:
            tmp.rename(dst)
        return r.returncode

    with ThreadPoolExecutor(args.workers) as ex:
        codes = list(ex.map(get, jobs))
    print(f"done: {codes.count(0)}/{len(jobs)} ok")


if __name__ == "__main__":
    main()
