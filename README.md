# Automatic winding chains for the villa spiral fitter

This repository builds automatic relative-winding annotations for the
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) spiral fitter and measures what they do to the fit.

- **`autowind`** (`tools/autowind/`) reads a binary surface-mask OME-Zarr and an umbilicus. It writes
  `relative_windings.json`, with one chain per radial ray of sheet crossings and a cumulative winding count. The fitter
  loads the chains as unattached strips.
- **Export tools** (`autowind.export`) convert chains to and from the point-collection format that VC3D and the
  fitter read, so that a person can review and correct chains in VC3D.
- **A chain-weight anneal** for the fitter (`spiral/patches/`): the chain losses fix the winding count while the
  surface forms, then relax.
- **Evaluation code** (`eval/`) scores a fit against human meshes: wrong-winding area, seam switches, and CT-based
  label audits.
- **Notes** (`notes/`) hold the pre-registrations and results of each test, including the failures.

Scroll data is not included (see [Data](#data)).

## Results, with caveats

Each claim below links to the note that pre-registered it. Numbers are wrong-winding area on human-labelled
vertices, in percentage points (pp), method minus no-chain baseline. Negative is better.

- **PHerc1667 (Scroll 4), z 6500-8000: passed.** The no-raise recipe (M2) beat the baseline in all 6 seed
  combinations: inner wrong-winding area fell by 27.4 to 29.3 pp. This is about 86 % of the gain from human-oracle
  chains. M2 was the secondary arm, and its choice was informed by a dry run. The primary arm (M1) failed.
  See `notes/2026-10-22-confirmation-region.md`.
- **PHerc1667, whole scroll (7 bands, z 400-9000): provisional.** On the first seed pair, pooled over the 5 unseen
  bands, inner wrong area fell from 77.6 % to 35.0 %. The gain is small in the top band (z 8000-9000), where the
  fitter extrapolates the umbilicus. The second seed pair was still pending when this export was made.
  See `notes/2026-10-26-p1667-wholescroll-prereg.md` and `notes/2026-10-26-topband-umbilicus-prereg.md`.
- **Kelly's patch index is more accurate than our numbering on Scroll 4.** We scored both on the same vertices.
  Kelly's index on Stevens' patches is right on 95.9 % of co-covered vertices, and our numbering on 73.2 %. It
  numbers 68 % of the labelled inner surface correctly, and we number 49 %. Ours is better only at the scroll
  bottom (z 400-1500). Our real contributions are narrower: the chains raise the fitter's accuracy on covered
  vertices from 20 % to 66 %, and the fit numbers surface where no patch exists. See
  `notes/2026-10-27-kelly-headtohead-prereg.md`.
- **Tightly packed sheets (PHercParis4, Scroll 1): the plain recipe makes fits worse.** Where sheets touch, the
  mask merges them and the chains undercount. A pre-registered test on z 9000-10500 failed
  (`notes/2026-10-23-third-region-prereg.md`). A cut recipe improved the primary metric on z 7000-8500 by 7.9 to
  8.5 pp, but it failed the pre-registered limit on seam switches (`notes/2026-10-24-f6-confirm-prereg.md`).
- **Chain-weight anneal (test 13): passed on a fresh Paris4 region (z 10800-11800).** The primary metric on
  CT-audited labels improved by 2.0 to 3.9 pp in all 6 pairs. That is about a third of the effect seen in
  development, and about 6-11 % of the oracle's headroom. On audited labels, the non-primary windings got slightly
  worse. See `notes/2026-10-26-anneal-prereg.md` and `notes/2026-10-25-chain-loss-prereg.md` (test 12, which
  failed). For the port to the current villa fitter, see `notes/2026-10-26-fitter-head-port.md`.
- **Gates that stopped at the development stage:** label-free contact detection, fiber paths, dinovol features,
  newer surface models, and HercuViT did not separate merged sheets well enough to use. See the `*-gate.md` notes.
- **Structure ablation (Paris4 z 7000-8500, dev band, one seed per arm): linking matters most.** Each arm removes
  one property of the oracle chains. Share of the oracle's gain recovered: long linked pieces thinned to our sheets
  1.04, our breaks with correct steps 0.54, every sheet cut short 0.38, our automatic chains 0.25. Once pieces are
  linked, completeness adds nothing. This is a direction from one band, not a confirmed effect. See
  `notes/2026-10-28-structure-ablation-prereg.md` and `release/figures/structure_ablation_p4z7.png`.
- **Prepared but not run at export time:** the KP-chains test on Scroll 4 (`notes/2026-10-29-kp-chains-prereg.md`).
  A timed human-check session is planned. Only its instructions are included (`notes/2026-10-24-hitl-session-howto.md`).

Use `autowind` where the mask's sheet spacing is about 24 voxels or more. On tighter packing, first check step
precision on a labelled patch. The full guide is `release/AUTOWIND.md`.

## Install

You need Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repository> && cd <repository>
uv sync
uv run python -m autowind --help
```

`uv sync` installs `tools/` in editable mode, so `python -m autowind` works from any directory in the repository.

## Run

Build chains for one band of a mask. The mask can be a local path or an `s3://` URL, which is read anonymously.

```sh
uv run python -m autowind chains --mask <surface-mask OME-Zarr> --umbilicus <umbilicus.json> \
    --z0 6500 --z1 8000 --voxel-um 9.596 --raise off --out <run dir>
uv run python -m autowind presets          # the recorded presets (their paths follow the authors' data/ layout)
uv run python -m autowind chains --help    # all flags; the defaults are the confirmed PHerc1667 recipe
```

Cut chains at uncertain steps, as in the f6 recipe for tight packing:

```sh
uv run python -m autowind.postproc --run <run dir> --mode cut --out <cut run dir>
```

Export chains for VC3D or the fitter, and read edited files back:

```sh
uv run python -m autowind.export relative <run dir> --dst relative_windings.json
uv run python -m autowind.export relative <run dir> --split-perr 0.5 --dst relative_windings.json
uv run python -m autowind.export absolute <run dir> --offsets offsets.json --dst abs_winding.json
uv run python -m autowind.export import edited.json --dst chains_edited.json --against <run dir>
uv run python -m autowind.export check relative_windings.json
```

Tests:

```sh
uv run python tools/autowind/test_smoke.py    # synthetic spiral, about 20 s; the PHerc1667 case skips without data
uv run python tools/autowind/test_export.py   # needs an existing chain run under data/scratch; skips otherwise
```

Score a fit (needs the human meshes and reference caches; see `notes/2026-10-20-spiral-eval.md`):

```sh
uv run python eval/spiral_eval.py --help
```

## Layout

| Path | Contents |
|---|---|
| `tools/autowind/` | The chain CLI, post-processing, export and tests |
| `tools/` | The chain algorithm (`auto_windings.py`), its two small step models (`*.json`), the check viewer and helpers |
| `eval/` | Fit scoring, label audits, method tables, the Kelly head-to-head |
| `spiral/` | Driver scripts and arm configs (`arms/*.env`) for the villa fits, and the fitter patches (`patches/`) |
| `release/` | Figure scripts, the figures, `AUTOWIND.md`, the villa docs patch, frozen copies of scored code (`frozen/`), and the set builders and scorers of tests 13, the structure ablation and KP chains (`scored/`; their outputs stay private) |
| `sim/`, `forward/`, `train/`, `scripts/`, `cloud/` | Earlier work: a synthetic scroll simulator, a CT forward model, surface-model training and data scripts, pod setup |
| `notes/` | Pre-registrations and results, one dated note per piece of work |

Notes on this export:

- **`spiral/` is a record, not a turnkey pipeline.** `spiral/run_arm.sh` expects a villa checkout (commit
  0fb38c45 for the pinned fits), a fit recipe at `config/recipe.toml`, and helper stage scripts under `stages/`.
  The last two are not in this repository. The recipe overrides come from the published PHerc. 343 spiral-fit recipe.
- **Paths were rewritten.** This is a curated copy of a private working repository. In the exported files, paths
  such as `spiral/` and `$HOME` were rewritten, and pod IDs were replaced with `<pod-id>`. Because of this, the
  sha256 of some files differs from the value quoted in the notes. The quoted values refer to the original files.
- **Some referenced files are not here.** Some notes cite private notes, data under `data/`, or figures that were
  left out. Figures of PHercParis4 2026 data are left out because that data is under a publication reservation
  until 2027-06-25. The one exception is the structure-ablation chart, which shows chain points and scores only.

## Data

No scroll data is included: no CT volumes, masks, meshes, labels, or arrays derived from labels. The code reads public
Vesuvius Challenge data from `s3://vesuvius-challenge-open-data/` and Hugging Face, or from a local `data/` directory.
To get the data, see [scrollprize.org/data](https://scrollprize.org/data) and the data browser. Most data is licensed
CC BY-NC 4.0. The PHercParis4 2026 scans carry extra terms, including a publication reservation until 2027-06-25.

The figures in `release/figures/` that show CT data are derived from this data. They are licensed CC BY-NC 4.0, not
Apache-2.0 (see `NOTICE`).

## Licence

The code and notes are licensed under the Apache License 2.0 (see `LICENSE`). Third-party parts and the CC BY-NC
figures are listed in `NOTICE`.
