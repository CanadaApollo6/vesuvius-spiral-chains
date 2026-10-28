"""Load check for the KP / KPM sets through the fitter's own parsing code, CPU only, no torch import.
notes/2026-10-29-kp-chains-prereg.md section 2. Same procedure as data/scratch/structure_ablation/check_load.py.

  uv run python data/scratch/kp_chains/check_load.py

Executes load_point_collection and normalise_pcl_winding_annotations taken verbatim (ast source segments) from the
villa 0fb38c45 spiral point_collection.py used by the local fits (read-only), then applies fit_spiral.py's unattached
z-roi rule per band (longest id-sorted run with z in [Z0 - margin, Z1 + margin), margin = villa default 0; drop
< 2 points) and the strip rule (< 2 points dropped). Also runs tools/autowind/export.py check_doc (role relative).
Counts are compared with sets/manifest.json (kp: pieces/points; kpm: collections/points).
"""
import ast
import io
import json
import math
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SPIRAL = REPO / "data/scratch/anneal/local_probe/ext/villa/volume-cartographer/scripts/spiral"
sys.path.insert(0, str(REPO / "tools"))
from autowind.export import check_doc  # noqa: E402

head = subprocess.run(["git", "-C", str(SPIRAL), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head.startswith("0fb38c45"), head
src = (SPIRAL / "point_collection.py").read_text()
ns = {"json": json, "np": np, "os": __import__("os"), "Optional": __import__("typing").Optional,
      "Dict": dict, "Any": object}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("load_point_collection", "normalise_pcl_winding_annotations"):
        exec(compile(ast.get_source_segment(src, node), "point_collection.py", "exec"), ns)

MARGIN = 0
SD = REPO / "data/scratch/kp_chains/sets"
man = json.loads((SD / "manifest.json").read_text())
out = {"villa_head": head}
bad = []
for zs, mb in man["bands"].items():
    Z0, Z1 = mb["z"]
    for kind in ("kp", "kpm"):
        name = f"{kind}_z{Z0}"
        path = SD / name / "relative_windings.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            cols = ns["load_point_collection"](str(path))
        assert cols, buf.getvalue()
        kept = {}
        for pid, pcl in cols.items():
            for p in pcl["points"].values():
                p["zyx"] = np.array([p["p"][2], p["p"][1], p["p"][0]], dtype=np.float32)
            items = sorted(pcl["points"].items(), key=lambda kv: int(kv[0]))
            bs, be, rs = 0, 0, 0
            for i, (_, p) in enumerate(items):
                if Z0 - MARGIN <= p["zyx"][0] < Z1 + MARGIN:
                    if i + 1 - rs > be - bs:
                        bs, be = rs, i + 1
                else:
                    rs = i + 1
            if be - bs >= 2:
                pcl["points"] = dict(items[bs:be])
                kept[pid] = pcl
        ns["normalise_pcl_winding_annotations"](kept)
        n_pts = sum(len(p["points"]) for p in kept.values())
        ann_all = all(p["has_winding_annotations"] for p in kept.values())
        mono = all(all(b >= a for a, b in zip(w, w[1:])) for w in
                   ([q["winding_annotation"] for _, q in sorted(p["points"].items(), key=lambda kv: int(kv[0]))]
                    for p in kept.values()))
        finite = all(math.isfinite(q["winding_annotation"]) for p in kept.values() for q in p["points"].values())
        chk = check_doc(json.loads(path.read_text()), role="relative", filename=str(path))
        warn_kinds = sorted({w.split(": ", 1)[1].split(";")[0] for w in chk["warnings"]})
        exp = (mb["kp"]["pieces"], mb["kp"]["points"]) if kind == "kp" else (mb["kpm"]["collections"], mb["kpm"]["points"])
        r = dict(loaded_collections=len(cols), strips_kept=len(kept), points_kept=n_pts, all_annotated=ann_all,
                 wind_nondecreasing=mono, wind_finite=finite, loader_msg=buf.getvalue().strip(),
                 check_ok=chk["ok"], check_errors=chk["n_errors"], check_warnings=chk["n_warnings"],
                 warning_kinds=warn_kinds, manifest=list(exp), matches_manifest=(len(kept), n_pts) == exp)
        out[name] = r
        print(name, json.dumps(r), flush=True)
        if not (r["check_ok"] and r["matches_manifest"] and ann_all and finite and mono):
            bad.append(name)
(REPO / "data/scratch/kp_chains/check_load.json").write_text(json.dumps(out, indent=1))
print("LOAD CHECK", "FAIL " + str(bad) if bad else "OK")
sys.exit(1 if bad else 0)
