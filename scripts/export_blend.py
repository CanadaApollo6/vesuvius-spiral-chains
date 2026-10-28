#!/usr/bin/env python
"""Export the sheet instances of a synthetic scroll cell to a Blender .blend file.

Usage:
    uv run python scripts/export_blend.py CELL.zarr [--res 7p91|2p4] [--out OUT.blend] [--step 1]

Each instance id > 0 in ``instance_<res>`` becomes one mesh (marching cubes on its
binary mask), written as a binary PLY in a temp dir. A headless Blender run then
imports every PLY, names it ``sheet_<id>`` and gives it a distinct material colour.
"""

import argparse
import colorsys
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr
from skimage.measure import marching_cubes

MIN_VOXELS = 50

BLENDER_SCRIPT = r'''
import colorsys
import glob
import os
import sys

import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1:]
tmpdir, out_path = argv[0], argv[1]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

ply_files = sorted(glob.glob(os.path.join(tmpdir, "*.ply")))
n_imported = 0
for path in ply_files:
    sheet_id = int(os.path.splitext(os.path.basename(path))[0])
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=path)
    new_objs = [o for o in bpy.data.objects if o not in before]
    if not new_objs:
        raise RuntimeError("ply_import produced no object for " + path)
    obj = new_objs[0]
    obj.name = "sheet_%d" % sheet_id
    if obj.data is not None:
        obj.data.name = "sheet_%d_mesh" % sheet_id

    hue = (sheet_id * 0.618) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
    mat = bpy.data.materials.new(name="sheet_%d_mat" % sheet_id)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    mat.diffuse_color = (r, g, b, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    n_imported += 1

bpy.ops.wm.save_as_mainfile(filepath=out_path)
print("export_blend: imported %d meshes, saved %s" % (n_imported, out_path))
'''


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary little-endian PLY with float x/y/z vertices and triangle faces."""
    verts = np.asarray(verts, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    face_arr = np.empty(len(faces), dtype=face_dtype)
    face_arr["n"] = 3
    face_arr["idx"] = faces
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())
        f.write(face_arr.tobytes())


def mask_to_mesh(mask: np.ndarray, voxel_um: float, step: int):
    """Marching cubes on a binary mask. Returns (verts_xyz_mm, faces) or None if empty."""
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0:
        return None
    # Crop to the bounding box (plus one voxel of margin) for speed.
    lo = [max(int(a.min()) - 1, 0) for a in (zs, ys, xs)]
    hi = [int(a.max()) + 2 for a in (zs, ys, xs)]
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # 1-voxel zero pad so the surface is closed at the crop boundary.
    padded = np.pad(sub.astype(np.float32), 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step)
    # Remove the pad and move to global voxel coordinates (still z,y,x).
    verts = verts - 1.0 + np.asarray(lo, dtype=np.float64)
    # Scale to mm and reorder (z,y,x) -> (x,y,z).
    verts_xyz = (verts[:, ::-1] * (voxel_um / 1000.0)).astype(np.float32)
    # The axis reversal is a reflection, so flip winding to keep normals outward.
    faces = faces[:, ::-1]
    return verts_xyz, faces


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cell", help="path to a cell .zarr group")
    p.add_argument("--res", choices=["7p91", "2p4"], default="7p91", help="instance resolution (default 7p91)")
    p.add_argument("--out", default=None, help="output .blend path (default <cell_dir>/<cell_stem>_<res>.blend)")
    p.add_argument("--step", type=int, default=1, help="marching cubes step_size (default 1)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cell = Path(args.cell)
    out = Path(args.out) if args.out else cell.parent / f"{cell.stem}_{args.res}.blend"
    out = out.resolve()

    group = zarr.open_group(str(cell), mode="r")
    inst = np.asarray(group[f"instance_{args.res}"][...])
    cell_attrs = group.attrs["cell"]
    voxel_um = float(cell_attrs["vox_lo_um"] if args.res == "7p91" else cell_attrs["vox_hi_um"])

    tmpdir = Path(tempfile.mkdtemp(prefix="export_blend_"))
    n_meshes = 0
    try:
        ids = np.unique(inst)
        ids = ids[ids > 0]
        for sheet_id in ids:
            mask = inst == sheet_id
            if int(np.count_nonzero(mask)) < MIN_VOXELS:
                continue
            mesh = mask_to_mesh(mask, voxel_um, args.step)
            if mesh is None:
                continue
            verts, faces = mesh
            write_ply(tmpdir / f"{int(sheet_id)}.ply", verts, faces)
            n_meshes += 1

        if n_meshes == 0:
            print("export_blend: no instances with >= %d voxels; nothing to export" % MIN_VOXELS, file=sys.stderr)
            return 1

        script = tmpdir / "_import_plys.py"
        script.write_text(BLENDER_SCRIPT)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "blender", "-b", "--python-exit-code", "1",
            "-P", str(script), "--", str(tmpdir), str(out),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"export_blend: blender exited with {result.returncode}; temp dir kept at {tmpdir}", file=sys.stderr)
            return result.returncode
    except Exception:
        print(f"export_blend: failed; temp dir kept at {tmpdir}", file=sys.stderr)
        raise
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"meshes: {n_meshes}")
    print(f"output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
