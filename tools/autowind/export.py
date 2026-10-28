"""autowind export: chain output -> point-collection files that VC3D and the villa spiral fitter read, and back.

  # relative-winding document for VC3D review / the fitter (one collection per chain, review tags + risk colours)
  uv run python -m autowind.export relative <run dir> --dst <dataset>/relative_windings.json
  # conservative variant: cut chains at every step whose p_err >= 0.5 (bridged steps have p_err 1.0)
  uv run python -m autowind.export relative <run dir> --split-perr 0.5 --dst relative_windings.json
  # absolute-winding document from per-chain offsets (e.g. a human placed one global number per chain)
  uv run python -m autowind.export absolute <run dir> --offsets offsets.json --dst <dataset>/abs_winding.json
  # read a (possibly VC3D-edited) document back into chains.json form, and report what a human changed
  uv run python -m autowind.export import edited.json --dst chains_edited.json --against <run dir>
  # validate a document against the VC3D and fitter loader rules
  uv run python -m autowind.export check relative_windings.json

(run with PYTHONPATH=tools, or `cd tools`; stdlib only, no numpy needed.)

Format (one format for all three consumers; sources in notes/2026-10-26-chain-export.md):
  {"vc_pointcollections_json_version": "1", <extra top-level keys: kept by VC3D, ignored by the fitter>,
   "collections": {"<int id>": {"uuid": <uuid string>, "name": str, "color": [r, g, b],
       "metadata": {"winding_is_absolute": bool},      # VC3D treats a MISSING key as true (absolute)
       "tags": {str: str},                              # kept by VC3D, ignored by the fitter
       "points": {"<int id>": {"p": [x, y, z], "wind_a": number | null, "creation_time": int ms}}}}}
The fitter decides the role from the FILE NAME, not the metadata: abs_winding.json = absolute,
relative_windings.json = relative, same_windings.json = same-winding (all wind_a null), drawn_control_points.json.
It orders a collection's points by integer point id; a collection with some wind_a null loses those points.

Our chains: chains.json entries {z, theta, pts [[z, y, x], ...], wind [int], r, perr (per step), rl (per point)} in the
mask level-0 frame of the run (autowind_run.json). --affine maps xyz into another frame (3x4 or 4x4
"transformation_matrix", p' = A p + t, as villa's volume_transforms); import inverts it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid as uuidlib
from pathlib import Path

VERSION = "1"
ROLE_FILES = {"absolute": "abs_winding.json", "relative": "relative_windings.json", "same": "same_windings.json",
              "drawn": "drawn_control_points.json"}
NS = uuidlib.UUID("5f0b6d1e-7a52-4c1e-9a43-2d1f3c0a6b77")  # namespace for deterministic collection uuids
NAME_RE = re.compile(r"_z(-?\d+(?:\.\d+)?)_t(-?\d+(?:\.\d+)?)_(\d+)(?:_s(\d+))?$")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


# ------------------------------------------------------------------------------------------------- geometry
def load_affine(path: str | Path | None):
    if not path:
        return None
    M = json.loads(Path(path).read_text())
    M = M.get("transformation_matrix", M.get("matrix", M)) if isinstance(M, dict) else M
    M = [list(map(float, r)) for r in M][:3]
    if len(M) != 3 or any(len(r) != 4 for r in M):
        raise SystemExit(f"{path}: need a 3x4 or 4x4 matrix")
    return M


def apply(M, xyz):
    if M is None:
        return list(xyz)
    return [M[i][0] * xyz[0] + M[i][1] * xyz[1] + M[i][2] * xyz[2] + M[i][3] for i in range(3)]


def invert(M):
    """Inverse of the 3x4 affine (A|t) -> (A^-1 | -A^-1 t)."""
    if M is None:
        return None
    a = [r[:3] for r in M]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(det) < 1e-12:
        raise SystemExit("affine is singular")
    inv = [[(a[(j + 1) % 3][(i + 1) % 3] * a[(j + 2) % 3][(i + 2) % 3]
             - a[(j + 1) % 3][(i + 2) % 3] * a[(j + 2) % 3][(i + 1) % 3]) / det for j in range(3)] for i in range(3)]
    t = [r[3] for r in M]
    return [inv[i] + [-sum(inv[i][k] * t[k] for k in range(3))] for i in range(3)]


# ------------------------------------------------------------------------------------------------- inputs
def load_run(run: str | Path) -> tuple[list[dict], dict]:
    """Chains (chains.json, else rebuilt from relative_windings.json) and provenance of an autowind output dir."""
    run = Path(run)
    meta = {"run_dir": str(run)}
    rj = run / "autowind_run.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        o = r.get("options", {})
        meta.update(autowind=r.get("autowind"), mask=o.get("mask"), umbilicus=o.get("umbilicus"),
                    z0=o.get("z0"), z1=o.get("z1"), voxel_um=r.get("voxel_um", o.get("voxel_um")),
                    raise_applied=r.get("raise_applied"))
    if (run / "chains.json").exists():
        return json.loads((run / "chains.json").read_text()), meta
    if (run / "relative_windings.json").exists():  # older outputs: positions + windings only
        doc = json.loads((run / "relative_windings.json").read_text())
        return chains_from_doc(doc), meta
    raise SystemExit(f"{run}: no chains.json or relative_windings.json")


def _risk_colour(n_risky: int) -> list[float]:
    return [0.1, 0.8, 0.2] if n_risky == 0 else [0.95, 0.8, 0.1] if n_risky == 1 else [0.9, 0.15, 0.1]


def _segments(c: dict, split_perr: float | None) -> list[tuple[int, int]]:
    """[start, end) point ranges of a chain, cut at steps with p_err >= split_perr."""
    n = len(c["pts"])
    if split_perr is None:
        return [(0, n)]
    out, s = [], 0
    pe = c.get("perr") or [0.0] * (n - 1)
    for k, p in enumerate(pe):
        if p >= split_perr:
            out.append((s, k + 1))
            s = k + 1
    out.append((s, n))
    return [(a, b) for a, b in out if b - a >= 2]


# ------------------------------------------------------------------------------------------------- export
def chains_to_doc(chains: list[dict], meta: dict, *, prefix: str = "auto", affine=None, split_perr: float | None = None,
                  risk_perr: float = 0.5, min_points: int = 2, z_range=None, offsets: dict | None = None,
                  absolute: bool = False, decimals: int = 2) -> dict:
    """VC3D point-collection document from chains. offsets (absolute only): chain key -> integer added to wind."""
    cols, pid, ct = {}, 1, 1
    src = str(meta.get("run_dir", ""))
    for ci, c in enumerate(chains, 1):
        if z_range and not (z_range[0] <= c["z"] < z_range[1]):
            continue
        base = f"{prefix}_z{int(c['z'])}_t{c['theta']:g}_{ci}"
        off = 0
        if absolute:
            key = next((k for k in (base, str(ci)) if k in offsets), None)
            if key is None:
                continue
            off = int(offsets[key])
        segs = _segments(c, split_perr)
        for si, (a, b) in enumerate(segs):
            if b - a < min_points:
                continue
            name = base if len(segs) == 1 and (a, b) == (0, len(c["pts"])) else f"{base}_s{si}"
            w0 = 0 if absolute else c["wind"][a]
            pe = (c.get("perr") or [])[a:b - 1]
            risky = [k for k, p in enumerate(pe) if p >= risk_perr]
            pts = {}
            for k in range(a, b):
                z, y, x = c["pts"][k]
                pts[str(pid)] = {"p": [round(v, decimals) for v in apply(affine, (x, y, z))],
                                 "wind_a": int(c["wind"][k] - w0 + off), "creation_time": ct}
                pid += 1
                ct += 1
            tags = {"autowind_chain": str(ci), "z": f"{c['z']:g}", "theta_deg": f"{c['theta']:g}",
                    "first_point": str(a), "n_steps": str(b - a - 1), "n_risky_steps": str(len(risky)),
                    "risky_steps": ",".join(map(str, risky)),
                    "max_perr": f"{max(pe):.3f}" if pe else "nan", "source_run": src}
            if absolute:
                tags["abs_offset"] = str(off)
            cols[str(len(cols) + 1)] = {
                "uuid": str(uuidlib.uuid5(NS, f"{src}|{name}|{'abs' if absolute else 'rel'}")),
                "name": name, "points": pts, "metadata": {"winding_is_absolute": bool(absolute)},
                "color": _risk_colour(len(risky)), "tags": tags}
    head = {"vc_pointcollections_json_version": VERSION,
            "autowind_export": {"tool": "tools/autowind/export.py", "role": "absolute" if absolute else "relative",
                                "source": meta, "affine": affine, "split_perr": split_perr, "risk_perr": risk_perr,
                                "point_order": "point id ascending = chain order outward (wind_a non-decreasing)",
                                "frame": "chain mask level-0 voxels, p = [x, y, z]" if affine is None
                                else "affine(chain mask level-0 voxels), p = [x, y, z]"}}
    head["collections"] = cols
    return head


# ------------------------------------------------------------------------------------------------- import
def chains_from_doc(doc: dict, affine=None) -> list[dict]:
    """Chains (chains.json schema subset + name/uuid/tags) from a point-collection document, points in id order.
    affine: the export affine; its inverse is applied so points return to the chain frame."""
    inv = invert(affine)
    out = []
    for cid in sorted(doc["collections"], key=int):
        col = doc["collections"][cid]
        items = sorted(col.get("points", {}).items(), key=lambda kv: int(kv[0]))
        pts, wind = [], []
        for _, p in items:
            x, y, z = apply(inv, p["p"])
            pts.append([z, y, x])
            w = p.get("wind_a")
            wind.append(None if w is None else (int(w) if float(w).is_integer() else float(w)))
        tags = col.get("tags", {}) or {}
        m = NAME_RE.search(col.get("name", ""))
        z = float(tags["z"]) if "z" in tags else float(m.group(1)) if m else (pts[0][0] if pts else float("nan"))
        th = float(tags["theta_deg"]) if "theta_deg" in tags else float(m.group(2)) if m else float("nan")
        out.append({"z": z, "theta": th, "pts": pts, "wind": wind, "name": col.get("name"), "uuid": col.get("uuid"),
                    "tags": tags, "winding_is_absolute": (col.get("metadata") or {}).get("winding_is_absolute")})
    return out


def diff_against(edited: list[dict], orig: list[dict], tol: float = 0.05, move_tol: float = 3.0) -> dict:
    """What changed between exported (orig, whole chains) and edited collections, matched by the autowind_chain tag
    (else the name). A point within tol of an original point is the same point; within move_tol, moved; else added."""
    by_ci = {}
    for e in edited:
        m = NAME_RE.search(e.get("name") or "")
        ci = (e.get("tags") or {}).get("autowind_chain") or (m.group(3) if m else None)
        by_ci.setdefault(ci, []).append(e)
    rep = dict(chains=len(orig), collections_in=len(edited), chains_missing=0, chains_unchanged=0, chains_edited=0,
               points_moved=0, points_added=0, points_removed=0, steps_changed=0, new_collections=0, edited_chains=[])
    for ci, c in enumerate(orig, 1):
        es = by_ci.pop(str(ci), None)
        if not es:
            rep["chains_missing"] += 1
            continue
        moved = added = sc = 0
        matched = set()
        for e in es:
            idx = {}  # original point index -> edited wind
            for p, w in zip(e["pts"], e["wind"]):
                d, k = min((math.dist(p, q), k) for k, q in enumerate(c["pts"]))
                if d <= tol:
                    idx[k] = w
                    matched.add(k)
                elif d <= move_tol:
                    moved += 1
                    matched.add(k)
                else:
                    added += 1
            for k in range(len(c["pts"]) - 1):
                if k in idx and k + 1 in idx and idx[k] is not None and idx[k + 1] is not None:
                    sc += int((idx[k + 1] - idx[k]) != (c["wind"][k + 1] - c["wind"][k]))
        removed = len(c["pts"]) - len(matched)
        changed = bool(moved or added or removed or sc or len(es) > 1)
        rep["chains_edited" if changed else "chains_unchanged"] += 1
        rep["points_moved"] += moved
        rep["points_added"] += added
        rep["points_removed"] += removed
        rep["steps_changed"] += sc
        if changed:
            rep["edited_chains"].append(dict(chain=ci, collections=len(es), moved=moved, added=added,
                                             removed=removed, steps_changed=sc))
    rep["new_collections"] = sum(len(v) for v in by_ci.values())
    return rep


# ------------------------------------------------------------------------------------------------- check
def check_doc(doc: dict, role: str | None = None, filename: str | None = None) -> dict:
    """Problems a VC3D load (core/src/PointCollections.cpp) or a fitter load (spiral-fitting/point_collection.py,
    fit_spiral.py) would hit or silently change. errors = would fail or lose data; warnings = loads, but differently
    from what the role intends."""
    err, warn = [], []
    if doc.get("vc_pointcollections_json_version") != VERSION:
        err.append("vc_pointcollections_json_version must be the string \"1\"")
    cols = doc.get("collections")
    if not isinstance(cols, dict):
        return dict(ok=False, errors=err + ["collections must be an object"], warnings=warn)
    if filename and role:
        want = ROLE_FILES[role]
        if Path(filename).name != want:
            warn.append(f"the fitter infers the role from the file name; a {role} document should be named {want} "
                        f"(or passed with an explicit role)")
    pids, n_pts = set(), 0
    for cid, col in cols.items():
        tag = f"collection {cid}"
        if not str(cid).isdigit():
            err.append(f"{tag}: id must be a non-negative integer string")
        if "name" not in col:
            err.append(f"{tag}: missing name (VC3D requires it)")
        if "uuid" in col and not (isinstance(col["uuid"], str) and UUID_RE.match(col["uuid"])):
            err.append(f"{tag}: uuid present but not a valid uuid (VC3D rejects the whole file)")
        md = col.get("metadata")
        isabs = (md or {}).get("winding_is_absolute") if isinstance(md, dict) else None
        if not isinstance(isabs, bool):
            warn.append(f"{tag}: metadata.winding_is_absolute missing; VC3D will load it as ABSOLUTE")
        elif role == "absolute" and not isabs or role in ("relative", "same") and isabs:
            warn.append(f"{tag}: metadata.winding_is_absolute={isabs} disagrees with role {role}")
        if "tags" in col and not all(isinstance(v, str) for v in (col["tags"] or {}).values()):
            warn.append(f"{tag}: non-string tag values are dropped by VC3D")
        pts = col.get("points", {})
        if len(pts) < 2:
            warn.append(f"{tag}: fewer than 2 points (no strip / no relation for the fitter)")
        ws = []
        for pid, p in sorted(pts.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else -1):
            n_pts += 1
            if not str(pid).isdigit():
                err.append(f"{tag}: point id {pid!r} is not an integer")
                continue
            if pid in pids:
                warn.append(f"{tag}: point id {pid} repeats another collection's (VC3D ids are file-global)")
            pids.add(pid)
            xyz = p.get("p")
            if not (isinstance(xyz, list) and len(xyz) == 3 and all(isinstance(v, (int, float)) and
                                                                    math.isfinite(v) for v in xyz)):
                err.append(f"{tag}: point {pid} needs p = [x, y, z] finite numbers")
            w = p.get("wind_a")
            if w is not None and not (isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w)):
                err.append(f"{tag}: point {pid} wind_a must be a finite number or null")
                w = None
            ct = p.get("creation_time")
            if ct is not None and not (isinstance(ct, int) and not isinstance(ct, bool)):
                err.append(f"{tag}: point {pid} creation_time must be an integer (VC3D reads int64)")
            ws.append(w)
        ann = [w for w in ws if w is not None]
        if ann and len(ann) < len(ws):
            err.append(f"{tag}: mixed wind_a / null; the fitter strips the {len(ws) - len(ann)} null points")
        if role in ("relative", "absolute") and not ann and ws:
            err.append(f"{tag}: no wind_a; the fitter would treat it as same-winding")
        if role == "same" and ann:
            warn.append(f"{tag}: same-winding collections should carry wind_a null")
        if role == "relative" and ann and any(b < a for a, b in zip(ann, ann[1:])):
            warn.append(f"{tag}: wind_a decreases along point-id order (allowed; our chains are outward-monotone)")
    return dict(ok=not err, collections=len(cols), points=n_pts, errors=err[:50], n_errors=len(err),
                warnings=warn[:50], n_warnings=len(warn))


# ------------------------------------------------------------------------------------------------- CLI
def _write(doc: dict, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(doc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autowind.export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("relative", "absolute"):
        p = sub.add_parser(name)
        p.add_argument("run", help="autowind output dir (chains.json [+ autowind_run.json])")
        p.add_argument("--dst", required=True)
        p.add_argument("--affine", help="JSON 3x4/4x4 transformation_matrix applied to [x, y, z]")
        p.add_argument("--prefix", default="auto")
        p.add_argument("--min-points", type=int, default=2)
        p.add_argument("--z-range", type=float, nargs=2, help="keep chains with z0 <= z < z1 (chain frame)")
        p.add_argument("--split-perr", type=float, help="cut chains at steps with p_err >= this")
        p.add_argument("--risk-perr", type=float, default=0.5, help="steps at/above this are tagged risky / coloured")
        if name == "absolute":
            p.add_argument("--offsets", required=True, help="JSON {chain name or 1-based chain index: int offset}; "
                                                            "wind_abs = wind + offset; chains without one are skipped")
    p = sub.add_parser("import")
    p.add_argument("src")
    p.add_argument("--dst", required=True)
    p.add_argument("--affine", help="the affine used at export (inverted here)")
    p.add_argument("--against", help="original run dir: report what was edited")
    p = sub.add_parser("check")
    p.add_argument("src")
    p.add_argument("--role", choices=sorted(ROLE_FILES), help="default: from the file name")
    o = ap.parse_args(argv)

    if o.cmd in ("relative", "absolute"):
        chains, meta = load_run(o.run)
        offsets = json.loads(Path(o.offsets).read_text()) if o.cmd == "absolute" else None
        doc = chains_to_doc(chains, meta, prefix=o.prefix, affine=load_affine(o.affine), split_perr=o.split_perr,
                            risk_perr=o.risk_perr, min_points=o.min_points, z_range=o.z_range, offsets=offsets,
                            absolute=o.cmd == "absolute")
        _write(doc, o.dst)
        rep = check_doc(doc, o.cmd, o.dst)
        print(json.dumps({k: rep[k] for k in ("ok", "collections", "points", "n_errors", "n_warnings")}
                         | {"warnings": rep["warnings"][:3], "dst": o.dst}))
        return 0 if rep["ok"] else 1
    if o.cmd == "import":
        doc = json.loads(Path(o.src).read_text())
        ch = chains_from_doc(doc, load_affine(o.affine))
        Path(o.dst).write_text(json.dumps(ch))
        out = dict(collections=len(ch), points=sum(len(c["pts"]) for c in ch), dst=o.dst)
        if o.against:
            out["diff"] = diff_against(ch, load_run(o.against)[0])
            out["diff"]["edited_chains"] = out["diff"]["edited_chains"][:20]
        print(json.dumps(out, indent=1))
        return 0
    if o.cmd == "check":
        doc = json.loads(Path(o.src).read_text())
        role = o.role or {v: k for k, v in ROLE_FILES.items()}.get(Path(o.src).name)
        rep = check_doc(doc, role, o.src)
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
