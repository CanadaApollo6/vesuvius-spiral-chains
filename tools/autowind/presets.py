"""Presets: the inputs behind our published chain sets. Each preset only fills CLI defaults; explicit flags win.

Crop checkpoint dirs of presets are read-only sources (--crops-ro): crops that are not cached there are solved into
<out>/crops, so the frozen crop sets behind the October entry are never modified.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AW = REPO / "data" / "scratch" / "auto_windings"

PRESETS = {
    # PHerc0139 volume 20250728140407 (9.362 um), band 1. m7 streamed from the open-data bucket (anonymous S3);
    # umbilicus = the published 2.399 um umbilicus mapped with the published affine (exactly as eval/spiral_eval).
    # Chain sets: band_m7_ray_c (--raise off), band_m7_ray_gmxc (--raise gm, the band-1 recipe).
    "p0139_band1": dict(
        mask=("s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/"
              "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
        umbilicus=REPO / "data" / "eval" / "p0139_umbilicus" / "umbilicus_20260102150214.json",
        umbilicus_transform=REPO / "data" / "eval" / "surface_labels_2026_07_sample" / "transform_0139_2p399_to_9p362.json",
        z0=3000, z1=4500, voxel_um=9.362,
        work=AW, crops_ro=[AW / "band_m7" / "crops"],
        grad_mag=REPO / "data" / "p0139" / "inputs" / "PHerc0139_grad_mag.ome.zarr",
        grad_mag_extra=[AW / "lasagna_z4500" / "PHerc0139_grad_mag.ome.zarr"],
        shape_hint=(20974, 6621, 6621),
    ),
    # PHerc1667 volume 20251217075048 level 2 (9.596 um), the pre-registered confirmation band. Local m7 mask
    # (spiral/predict_band.py), human-mesh umbilicus in the fit frame.
    # Chain sets: band_m7_p1667_ray_c (--raise off, confirmed M2), band_m7_p1667_ray_gmxc (--raise gm, failed M1).
    "p1667": dict(
        mask=REPO / "data" / "p1667" / "masks" / "m7_L2_th0.2.zarr",
        umbilicus=REPO / "data" / "eval" / "p1667" / "umbilicus_p1667_L2.json",
        umbilicus_transform=None,
        z0=6500, z1=8000, voxel_um=9.596,
        work=AW / "p1667", crops_ro=[AW / "p1667" / "band_m7_p1667" / "crops"],
        grad_mag=REPO / "data" / "p1667" / "inputs" / "PHerc1667_grad_mag.ome.zarr",
        grad_mag_extra=[],
        shape_hint=(9269, 3808, 3808),
    ),
}
