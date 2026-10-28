"""autowind.export smoke / round-trip tests (seconds; stdlib + numpy for the fitter-loader check).

  uv run --with pytest pytest -q tools/autowind/test_export.py
  uv run python tools/autowind/test_export.py           # same tests without pytest

Input: an existing chain output, data/scratch/ink_lowband/autowind_z4500 (PHerc1667 20251217075048 L2, z 4500-6000,
580 chains, made by the autowind CLI). Skipped if absent.
1. relative export passes check_doc, and point-for-point equals the run's own relative_windings.json (same ids,
   positions, wind_a up to the per-chain base) -> the export changes metadata only, not the constraints.
2. the fitter's loader (load_point_collection + normalise_pcl_winding_annotations, taken verbatim by AST from the
   latest villa clone, no villa import) loads it: every collection fully annotated, ids in chain order.
3. a VC3D load -> save emulation (keys exactly as core/src/PointCollections.cpp to_json/from_json) then import
   returns the original chains (z, theta, wind exact; positions within rounding).
4. affine export + import with the same affine returns the original chains.
5. absolute export with offsets; split-perr export never contains a step at/above the cut.
6. a simulated human edit (one step changed, one point deleted) is reported by diff_against.
"""

import ast
import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from autowind import export as ex  # noqa: E402

RUN = REPO / "data" / "scratch" / "ink_lowband" / "autowind_z4500"
LOADERS = [REPO / "data" / "untrusted" / "villa_formats_1026" / "villa" / "spiral-fitting" / "point_collection.py",
           REPO / "data" / "ext" / "villa" / "spiral-fitting" / "point_collection.py"]


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP", msg)
        raise _Skip()


class _Skip(Exception):
    pass


def _run():
    if not (RUN / "chains.json").exists():
        _skip(f"{RUN} absent")
    return ex.load_run(RUN)


def _same_chains(a, b, tol=0.006):
    assert len(a) == len(b), (len(a), len(b))
    for ca, cb in zip(a, b):
        assert ca["z"] == cb["z"] and ca["theta"] == cb["theta"]
        wa = [w - ca["wind"][0] for w in ca["wind"]]
        wb = [w - cb["wind"][0] for w in cb["wind"]]
        assert wa == wb
        assert len(ca["pts"]) == len(cb["pts"])
        for p, q in zip(ca["pts"], cb["pts"]):
            assert max(abs(u - v) for u, v in zip(p, q)) <= tol, (p, q)


def vc3d_resave(doc):
    """What VC3D writes after loading `doc` and saving it unchanged (PointCollections.cpp loadFromJSON/saveToJSON)."""
    out = {k: v for k, v in doc.items() if k not in ("vc_pointcollections_json_version", "collections")}
    out["vc_pointcollections_json_version"] = "1"
    cols = {}
    for cid, c in doc["collections"].items():
        pts = {}
        for pid, p in c["points"].items():
            q = {"p": [float(v) for v in p["p"]], "creation_time": int(p.get("creation_time", 0)),
                 "wind_a": None if p.get("wind_a") is None else float(p["wind_a"])}
            if p.get("links"):
                q["links"] = p["links"]
            if p.get("fiber_dir") in ("h", "v"):
                q["fiber_dir"] = p["fiber_dir"]
            pts[pid] = q
        md = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        isabs = md.get("winding_is_absolute") if isinstance(md.get("winding_is_absolute"), bool) else True
        n = {"uuid": c.get("uuid", ""), "name": c["name"], "points": pts, "metadata": {"winding_is_absolute": isabs},
             "color": c.get("color", [0.2, 0.8, 1.0])}
        tags = {k: v for k, v in (c.get("tags") or {}).items() if isinstance(v, str)}
        if tags:
            n["tags"] = tags
        if c.get("windings_linked"):
            n["windings_linked"] = c["windings_linked"]
        cols[cid] = n
    out["collections"] = cols
    return json.loads(json.dumps(out))


def fitter_loader():
    src = next((p for p in LOADERS if p.exists()), None)
    if src is None:
        _skip("no villa spiral-fitting/point_collection.py clone")
    import numpy as np
    tree = ast.parse(src.read_text())
    want = {"load_point_collection", "normalise_pcl_winding_annotations"}
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {f.name for f in fns} == want, f"loader functions not found in {src}"
    import typing
    ns = {"json": json, "np": np, "os": __import__("os"), **{k: getattr(typing, k) for k in typing.__all__}}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(src), "exec"), ns)
    return ns["load_point_collection"], ns["normalise_pcl_winding_annotations"], src


def test_relative_matches_run_and_checks():
    chains, meta = _run()
    doc = ex.chains_to_doc(chains, meta)
    rep = ex.check_doc(doc, "relative", "relative_windings.json")
    assert rep["ok"] and rep["n_warnings"] == 0, rep
    orig = json.loads((RUN / "relative_windings.json").read_text())["collections"]
    assert list(doc["collections"]) == list(orig)
    for cid, col in doc["collections"].items():
        o = orig[cid]
        assert col["name"] == o["name"] and list(col["points"]) == list(o["points"])
        ow = [p["wind_a"] for p in o["points"].values()]
        for (pid, p), q, w in zip(col["points"].items(), o["points"].values(), ow):
            assert p["p"] == q["p"] and p["wind_a"] == w - ow[0]
        assert col["metadata"] == {"winding_is_absolute": False}
    print("relative export == run relative_windings.json:", rep["collections"], "collections,", rep["points"], "points")


def test_fitter_loader_reads_it():
    chains, meta = _run()
    load, normalise, src = fitter_loader()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "relative_windings.json"
        f.write_text(json.dumps(ex.chains_to_doc(chains, meta)))
        cols = load(str(f))
    assert cols is not None and len(cols) == len(chains)
    normalise(cols)
    assert all(c["has_winding_annotations"] for c in cols.values())
    for c, ch in zip((cols[k] for k in sorted(cols)), chains):
        w = [p["winding_annotation"] for _, p in sorted(c["points"].items())]
        assert w == [v - ch["wind"][0] for v in ch["wind"]]
    print("fitter loader OK:", src.relative_to(REPO))


def test_vc3d_resave_roundtrip():
    chains, meta = _run()
    doc = ex.chains_to_doc(chains, meta)
    re = vc3d_resave(doc)
    assert ex.check_doc(re, "relative")["ok"]
    assert re["autowind_export"]["role"] == "relative"  # VC3D keeps extra top-level keys
    back = ex.chains_from_doc(re)
    _same_chains(chains, back)
    assert all(c["tags"].get("autowind_chain") for c in back)
    print("VC3D resave -> import == chains")


def test_affine_roundtrip():
    chains, meta = _run()
    M = [[0.5, -3.2, 0.01, 11915.4], [-3.2, -0.5, -0.23, 14762.7], [0.23, 0.03, -3.28, 37027.5]]  # S4-like rotation
    doc = ex.chains_to_doc(chains[:50], meta, affine=M, decimals=4)
    back = ex.chains_from_doc(doc, affine=M)
    _same_chains(chains[:50], back, tol=0.01)
    print("affine export/import OK")


def test_absolute_and_split():
    chains, meta = _run()
    names = [f"auto_z{int(c['z'])}_t{c['theta']:g}_{i}" for i, c in enumerate(chains, 1)]
    offs = {names[0]: 11, "2": 20}
    doc = ex.chains_to_doc(chains, meta, absolute=True, offsets=offs)
    assert len(doc["collections"]) == 2 and ex.check_doc(doc, "absolute", "abs_winding.json")["ok"]
    c1 = doc["collections"]["1"]
    assert [p["wind_a"] for p in c1["points"].values()] == [w + 11 for w in chains[0]["wind"]]
    assert c1["metadata"]["winding_is_absolute"] is True
    doc = ex.chains_to_doc(chains, meta, split_perr=0.5)
    n_risky = sum(int(p >= 0.5) for c in chains for p in c["perr"])
    back = ex.chains_from_doc(doc)
    by = {}
    for c in back:
        by.setdefault(int(c["tags"]["autowind_chain"]), []).append(c)
    for ci, segs in by.items():
        ch = chains[ci - 1]
        for s in segs:
            a = int(s["tags"]["first_point"])
            assert all(p < 0.5 for p in ch["perr"][a:a + len(s["pts"]) - 1])
    print(f"absolute OK; split at {n_risky} risky steps -> {len(back)} collections from {len(chains)} chains")


def test_diff_reports_human_edit():
    chains, meta = _run()
    doc = ex.chains_to_doc(chains, meta)
    ed = copy.deepcopy(doc)
    col = ed["collections"]["3"]
    ids = sorted(col["points"], key=int)
    for pid in ids[5:]:
        col["points"][pid]["wind_a"] += 1      # human: step 4->5 is one more winding
    del col["points"][ids[-1]]                 # human: last point is off-sheet
    rep = ex.diff_against(ex.chains_from_doc(vc3d_resave(ed)), chains)
    assert rep["chains_edited"] == 1 and rep["edited_chains"][0]["chain"] == 3, rep
    assert rep["edited_chains"][0]["steps_changed"] == 1 and rep["edited_chains"][0]["removed"] == 1, rep
    print("diff:", {k: rep[k] for k in ("chains_unchanged", "chains_edited", "steps_changed", "points_removed")})


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except _Skip:
                pass
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:400])
    sys.exit(1 if fails else 0)
