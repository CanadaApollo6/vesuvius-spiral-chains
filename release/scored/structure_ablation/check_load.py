"""Load check for the structure-ablation sets through the fitter's own parsing code, CPU only, no torch import.

  uv run python data/scratch/structure_ablation/check_load.py

Executes load_point_collection and normalise_pcl_winding_annotations taken verbatim (ast source segments) from the
villa 0fb38c45 spiral point_collection.py used by the local fits (read-only), then applies fit_spiral.py's unattached
z-roi rule (longest id-sorted run with z in [z_begin - patch_loss_z_margin, z_end + margin), drop < 2 points;
margin = villa default 0, not in recipe.toml) and the strip rule (< 2 points dropped). Also runs
tools/autowind/export.py check_doc (role relative). Reference sets (oracle, p4z7_f6) are checked the same way.
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

Z0, Z1, MARGIN = 7000, 8500, 0
SETS = {"oracle": "data/scratch/f6_screen/confirm/windings/band_oracle_p4z7/relative_windings.json",
        "p4z7_f6": "data/scratch/f6_screen/confirm/windings/p4z7_f6/relative_windings.json",
        **{s: f"data/scratch/structure_ablation/sets/{s}/relative_windings.json" for s in ("sa_A1", "sa_A2", "sa_A3")}}
man = json.loads((REPO / "data/scratch/structure_ablation/sets/manifest.json").read_text())
out = {"villa_head": head}
for name, rel in SETS.items():
    path = REPO / rel
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
               ([q["winding_annotation"] for _, q in sorted(p["points"].items())] for p in kept.values()))
    finite = all(math.isfinite(q["winding_annotation"]) for p in kept.values() for q in p["points"].values())
    chk = check_doc(json.loads(path.read_text()), role="relative", filename=str(path))
    warn_kinds = sorted({w.split(": ", 1)[1].split(";")[0] for w in chk["warnings"]})
    r = dict(loaded_collections=len(cols), strips_kept=len(kept), points_kept=n_pts, all_annotated=ann_all,
             wind_nondecreasing=mono, wind_finite=finite, loader_msg=buf.getvalue().strip(),
             check_ok=chk["ok"], check_errors=chk["n_errors"], check_warnings=chk["n_warnings"], warning_kinds=warn_kinds)
    if name in man["stats"]:
        st = man["stats"][name]
        r["matches_manifest"] = (len(kept) == st["pieces"] and n_pts == st["points"])
    out[name] = r
    print(name, json.dumps(r))
(REPO / "data/scratch/structure_ablation/check_load.json").write_text(json.dumps(out, indent=1))
bad = [n for n in SETS if n.startswith("sa_") and not (out[n]["check_ok"] and out[n]["matches_manifest"]
                                                         and out[n]["all_annotated"] and out[n]["wind_finite"])]
print("LOAD CHECK", "FAIL " + str(bad) if bad else "OK")
sys.exit(1 if bad else 0)
