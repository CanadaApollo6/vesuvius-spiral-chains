# Reproducing the numbers in SUBMISSION_DRAFT.md

`release/reproduce.sh` rebuilds both benchmarks, runs the public m7 checkpoint, and prints every headline number.
You need only this repo: no `data/` directory, no scan downloads. Crops are streamed from the public buckets.

    bash release/reproduce.sh [stage ...]   # default: all; stages always run in pipeline order; DRY_RUN=1 prints commands only

## Prerequisites
- Linux, `uv`, `curl`, `unzip`, a C++ compiler (for the Betti-Matching-3D build in the Kaggle metric env).
- A CUDA GPU with at least 16 GB for stages `m7` and `size` (m7 at 1.18x peaks at 4.0 GiB per 256^3 block; the script checks for 16 GB).
- Disk, all under `data/`: about 6 GB (PHerc0139 set 2.6 GB, meshes 0.95 GB, hf2um 0.35 GB, m7 checkpoint 0.8 GB, metric env 0.6 GB, predictions under 1 GB), plus the uv cache for torch and nnunetv2.
- Network: the PHerc0139 build reads at most 17.5 GB of zarr chunks (upper bound from the build notes).
- On this machine the GPU stages write `~/.gpu-leases/vss-reproduce.json` and remove it on exit; without that directory the lease step does nothing. `HF_TOKEN` is optional (public repo) and is never printed.

## Stages
| stage | what it does | yields |
|---|---|---|
| `setup` | downloads `scrollprize/surface_m7_nnunet`; builds `data/metric/.venv` from the Kaggle metric zip | tools only |
| `bench` | builds the PHerc0139 winding set and the hf2um set; crops the bucket m7 predictions | 87 sites, 24,972 pairs, 104 crops (36 / 40 / 28) |
| `m7` (GPU) | m7 on both sets: PHerc0139 at spacing 1.1836 and 1.0, hf2um with and without the 7.91 um declaration | masks for every m7 row |
| `size` (GPU) | params, time and peak memory for one 256^3 block | section 5 size figures |
| `score` | patch graph on each mask set, GT row, hf2um official score and compressed separation | tables in sections 1 and 2 |
| `paired` | paired site bootstrap, certifier, GT contact undercount | +2.08 pp [1.26, 2.93], 39.7 %, AUC 0.76 |
| `figs` | `release/fig_classes.py`, two `release/fig_compressed.py` pair figures (`FIG_OUT`, default `release/figures`) | PNGs |
| `report` | compact table read from the CSV/JSON outputs (always printed last) | all of the above |

Intermediate results are in `data/preds/`; the report inputs are in `data/reproduce/`; logs are in `data/scratch/reproduce/`. A finished step is skipped on rerun unless `FORCE=1`.

## Rows that need our own checkpoints
m7d300 and real350 are not public. Set `M7D300=<nnU-Net trainer dir>` (and optionally `REAL350=...`) to add their rows, the m7d300 + m7 ensemble row, the m7d300 certifier numbers and both figure types. Without them those steps are skipped with a message, and the certifier runs on the public m7 masks, so its numbers differ from the draft.

## Expected runtimes (RTX 3090, 20 cores; from file times of the original runs, so approximate)
- `bench`: PHerc0139 build about 13 min with 6 workers, pair freeze about 17 min; the mining and hf2um download times were not recorded.
- `m7`: about 20 min per PHerc0139 spacing; hf2um inference about 10 min.
- `score`: patch graphs 10-40 min per mask set, run in parallel; hf2um official score about 25 s per crop (roughly 45 min per model), run in parallel; separation a few minutes.
- `paired`, `figs`, `report`: about 10 s together.

## Known gaps
- Crop selection is seeded, but a rebuild from scratch may not give exactly the same 87 sites; check the counts printed by `report`.
