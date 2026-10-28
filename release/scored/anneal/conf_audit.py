"""Test 13 confirmation stage 1 (notes/2026-10-26-anneal-prereg.md section 3d): the label audit of
eval/winding_audit.py (unchanged), applied as a LABEL-ONLY check to the fresh Paris4 band z 10800-11800.
winding_audit.py refuses that band through PRESETS["PHercParis4"]["forbidden_z"]; this wrapper clears that entry in
this process only (the code keys Paris4 handling on the preset name, so the preset is edited in memory rather than
copied under a new name). Inputs: organiser labels (labels_w0701.npz), CT, m7 mask, grad_mag. No fit, no chains.
  uv run python data/scratch/anneal/conf_audit.py extract --scroll PHercParis4 --z0 10800 --z1 11800 \
      --out data/scratch/label_audit/paris4/conf_z10800
  uv run python data/scratch/anneal/conf_audit.py analyse --out data/scratch/label_audit/paris4/conf_z10800 --anchor 30 --outward-ccw
  uv run python data/scratch/anneal/conf_audit.py report  --out data/scratch/label_audit/paris4/conf_z10800 --anchor 30
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval"))
import winding_audit as W  # noqa: E402

W.PRESETS["PHercParis4"]["forbidden_z"] = []
print("conf_audit: PHercParis4 forbidden_z cleared in this process (label-only audit of z 10800-11800)", flush=True)
if __name__ == "__main__":
    W.main()
