"""Anneal switch decomposition (test 13, notes/2026-10-26-anneal-prereg.md): data/scratch/f7_switch/switch_diag.py
unchanged, with one extra band: the test-13 arms (same chains p4z7_f6) plus cl_annz7 (test 12) and f6z_ms1 / f6z_ms2.
  uv run python data/scratch/anneal/switch_diag_an.py --band z7000_an --out data/scratch/anneal/switch_diag_z7000_an.json
Reported: the 'dchain' rows (local switch rate per 1000 matched pairs at < 16 and 16-32 vox from a kept chain point)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data/scratch/f7_switch"))
import switch_diag as SD  # noqa: E402
VD = {a: "data/scratch/anneal/vertex" for a in ["an_f03z7", "an_e03z7"]}
VD["cl_annz7"] = "data/scratch/chain_loss/vertex"
ARMS = [a for a, d in VD.items() if (REPO / d / f"{a}.npz").exists()]
SD.BANDS["z7000_an"] = dict(SD.BANDS["z7000"], vd={**SD.BANDS["z7000"]["vd"], **{a: VD[a] for a in ARMS}},
                            methods=ARMS, others=["f6z_ms1", "f6z_ms2"],
                            cut="data/scratch/f6_screen/confirm/windings/p4z7_f6")
if __name__ == "__main__":
    SD.main()
