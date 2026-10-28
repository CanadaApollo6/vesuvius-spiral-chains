# Chain export for human correction (VC3D, spiral fitter, Gill & Tran format), 2026-10-26

New files only: `tools/autowind/export.py` and `tools/autowind/test_export.py`. No frozen file was touched
(`auto_windings.py`, `postproc.py`, the `cli.py`/`core.py` shas in the preregs).

## Result: all three targets use one format

All three targets read the **VC3D point-collection JSON v1**. Our `relative_windings.json` was already valid for the
fitter. The three targets differ only in role conventions and metadata.

| Consumer | What it reads | Source (villa main @ 322cb08, 2026-10-24, cloned to data/untrusted/villa_formats_1026) |
|---|---|---|
| (c) spiral fitter | `{"vc_pointcollections_json_version": "1", "collections": {"<int>": {"name", "points": {"<int>": {"p": [x,y,z], "wind_a", "creation_time"}}, "metadata", "color", "uuid"?}}}`. **The role comes from the file name, not the metadata:** `abs_winding.json` = absolute, `relative_windings.json` = relative, `same_windings.json` = same-winding, `drawn_control_points.json`. Points are ordered by integer id. If every point has `wind_a` null, the collection is same-winding (wind 0). If only some are null, those points are dropped. A point within 2.5 vox of a patch attaches to it, and `between_patches__A__B` names pin the patch pair. | [spiral-fitting/point_collection.py](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/point_collection.py) `load_point_collection`, `normalise_pcl_winding_annotations`; [fit_session.py](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/fit_session.py) `PCL_ROLE_CONVENTIONS`; `fit_spiral.py` `stamp_loaded_pcl_metadata` ("Absolute-winding status is determined solely by the source file"); README "Point-to-patch linking"; [spiral-input README](https://scrollprize.org/data/datasets/spiral-input-PHercParis4-README.md) |
| (c') fit service (VC3D Spiral panel, live edits) | Same document. A *replacement* upload of one collection needs point ids contiguous from 0, `creation_time` strictly increasing, and a finite `wind_a` on every point for relative. It rejects `links` / `windings_linked`. | `spiral-fitting/service_uploads.py` `_validate_replacement_document` |
| (b) VC3D | Same keys plus optional `uuid` (it must be a valid UUID or the **whole file is rejected**), `tags` (str→str), `anchor2d`, `autoFillMode`, `windings_linked`, and per point `links`, `fiber_dir`. Extra top-level keys survive a load/save. **A missing `metadata.winding_is_absolute` loads as `true`.** | [volume-cartographer/core/src/PointCollections.cpp](https://github.com/ScrollPrize/villa/blob/main/volume-cartographer/core/src/PointCollections.cpp) `to_json`/`from_json`/`loadFromJSON`; tutorial [VC3D § Winding Annotation](https://scrollprize.org/tutorial_VC3D) |
| (a) Gill & Tran winding-annotation tool ("PCU", Sept 2026, $500) | The same v1 document, written as an extra `relative_windings.json`: `metadata: {"winding_is_absolute": false, "generator": ...}`, `wind_a` as floats from 1, and no `creation_time`. Fit settings in their README: 10x unattached sampling, weights 10/10. | [github.com/Jashann/vesuvius-scrolling](https://github.com/Jashann/vesuvius-scrolling) `pcu/constraints.py:126 to_vc3d_json`, `exp/e63_merge_gold.py`, `data/README.md` |
| Kelly (Sept, for context) | The same format: `between_patches__A__B` two-point collections in `same_windings.json` (no `wind_a`) and `relative_windings.json` (0, delta). | [github.com/tf-kelly/scrolls](https://github.com/tf-kelly/scrolls) `phase/sc/convert.py:145-170` |

Villa's own machine writer (`connect_overlapping_patches.py`) emits `uuid4`, `metadata {"winding_is_absolute": false}`,
`color`, `creation_time 0` and `wind_a`. Our exporter follows the same pattern.

**Finding (no action on frozen code).** Our current `relative_windings.json` (written by frozen `auto_windings.write_json`)
has `"metadata": {}`. The fitter does not care, because the role comes from the file name. VC3D loads every chain as
**absolute**, though, and after a save it writes `winding_is_absolute: true`. The export fixes this without touching
the frozen writer.

**Undetermined:** VC3D's SpiralPointCollectionEdit builds its own replacement payload, so I did not check whether it
reorders points by `creation_time` or by id. Our export makes the two agree (both ascending outward), so either way the
order is the same.

## export.py

Stdlib only. Run from `tools/` (or with `PYTHONPATH=tools`).

```bash
cd tools
# relative document, one collection per chain (the run's relative_windings.json, plus metadata, uuid, tags, colours)
uv run python -m autowind.export relative ../data/scratch/ink_lowband/autowind_z4500 --dst /tmp/ds/relative_windings.json
# conservative: cut chains at steps with p_err >= 0.5 (bridged steps carry p_err 1.0)
uv run python -m autowind.export relative <run> --split-perr 0.5 --dst relative_windings.json
# into another frame (villa volume_transforms 3x4 matrix, p' = A p + t), e.g. L2 -> 2.399 um level 0 = 4 * identity
uv run python -m autowind.export relative <run> --affine l2_to_l0.json --dst relative_windings.json
# absolute: {"auto_z4512_t0_1": 11, "2": 20} (chain name or 1-based index -> offset); chains without one are skipped
uv run python -m autowind.export absolute <run> --offsets offsets.json --dst abs_winding.json
# after a human edits it in VC3D: back to chains.json form, plus a report of what changed
uv run python -m autowind.export import edited.json --dst chains_edited.json --against <run>
uv run python -m autowind.export check some_file.json [--role relative|absolute|same|drawn]
```

What each collection carries for review in VC3D:
- `tags`: autowind_chain, z, theta_deg, first_point, n_steps, n_risky_steps, risky_steps (step indices with p_err ≥
  `--risk-perr`, default 0.5), max_perr, source_run.
- `color`: green = no risky step, yellow = 1, red = ≥ 2.
- Deterministic `uuid5`, so a re-export keeps the same ids.
- `creation_time` = point index, strictly increasing.
- `metadata.winding_is_absolute` set to match the role.
- Top-level `autowind_export` provenance (source run, frame, affine), which VC3D keeps and the fitter ignores.

`import` matches collections back to chains by tag, or by name when VC3D dropped nothing. It inverts the affine and
reports chains edited, steps changed, and points moved, added or removed. That is what a HITL session needs to measure
human corrections.

The output is not a same-winding export. Chains are radial, so they assert no same-sheet relation between collections.
An honest same-winding output needs a global numbering first (see notes/2026-10-26-patch-numbering-scout.md).

## Test

`uv run --with pytest pytest -q tools/autowind/test_export.py` gives **6 passed in 0.55 s** (the same tests also run
without pytest). The input is `data/scratch/ink_lowband/autowind_z4500`: PHerc1667 L2, 580 chains, 18,531 points.
1. The relative export passes `check` with 0 errors and 0 warnings. It matches the run's own `relative_windings.json`
   point for point (ids, positions, `wind_a`), so only metadata changes.
2. The **fitter's own loader**, extracted verbatim by AST from villa main and run without importing villa, loads all
   580 collections fully annotated, with `wind_a` in chain order.
3. A VC3D load→save emulation (exact keys of PointCollections.cpp), then import, returns the original chains: z,
   theta and wind exact, positions within 0.006 vox.
4. Export and import through a Scroll-4-like rotation affine round-trip within 0.01 vox.
5. The absolute export with offsets is correct. `--split-perr 0.5` cuts at 1,415 risky steps and gives 1,690
   collections, none of which contains a step at or above the cut.
6. A simulated human edit (one step +1, one point deleted) is reported as exactly 1 chain edited, 1 step changed and 1
   point removed. The other 579 are unchanged.

Not tested: a real VC3D binary or a real fit. No fit was run.
