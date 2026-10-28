# 2026-10-16: eval harness, held-out pool, metric

## Held-out hard-region pool (found, local)
- Source: Jinho Jeong, https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic
  release `patches-v2` (2026-07-25). Listed on scrollprize.org community projects.
- 200 patches = 200-random seed-fixed sample of bruniss's `Dataset059_s1_s4_s5_patches_frangiedt`
  (1754 patches total, ~18 GB, https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/).
- Mix: 162 Scroll 1, 36 Scroll 4 (PHerc1667), 2 Scroll 5. Shapes 172³/236³/300³ (+2 at 170³).
  Labels are binary {0,1} (no ignore label 2 in the pool), LZW TIFF (needs `imagecodecs`).
- Local: `data/eval/jeong200/{imagesTr,labelsTr}` (tars kept in `_tars/`).
- **Resolution, verified by voxel matching: all patches are from the old 7.91 um DLS scans, not the 2025-26 BM18 scans.**
  - s4 patches == `dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr` level 0 (uint16, 53 keV, 7.91 um), r = 0.9999-1.0000 on 3/3 patches.
  - s1 patches == `.../Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr` (uint8), r ~0.97 on 3/3 (standardized copy, slightly different intensity mapping).
  - **Filename coords are the grid-cell origin; the patch starts at origin + (10,10,10)** (cells are patch+20, a 10-voxel margin trimmed).
  - No correlation with the 2.4 um S3 volumes at levels 1-3. No published transform from the old DLS scans to the new 2.4 / 1.13 um scans (Scroll 4 `transforms/` only has 3.24 um -> 7.91 um).
- Caveat (Jeong): patches likely overlap the public baselines' training data, so baseline numbers on them are optimistic. GT bands miss ~6.4% of sheet voxels on average (worst decile >14%).

## Official metric (built, sanity-checked)
- Kaggle dataset `sohier/vesuvius-metric-resources`, anonymous download works:
  `curl -L https://www.kaggle.com/api/v1/datasets/download/sohier/vesuvius-metric-resources`
  sha256 64d24044b7381dbb660a0e9f602ad0f8fd37d1935c09a8ee3b78088433910e68 (matches the hash pinned by tifxyz-doctor).
- Blend = 0.3 TopoScore (Betti matching, dims 0/1/2) + 0.35 SurfaceDice@2vox + 0.35 VOI score (1/(1+VOI), alpha 0.3, 26-conn).
- Isolated env `data/metric/.venv` (Python 3.11 + the zip's pinned wheels, offline). Betti-Matching-3D needs
  `CXXFLAGS="-include functional"` under GCC 16, and an absolute PATH to the venv so CMake picks 3.11:
  `CXXFLAGS="-include functional" PATH=$PWD/data/metric/.venv/bin:$PATH bash data/metric/topological-metrics-kaggle/scripts/build_betti.sh`
- Wrapper: `data/metric/.venv/bin/python eval/score.py --gt-dir ... --pred-dir ... --out x.csv`
- Sanity on 12 patches (6 s1, 6 s4): GT vs GT = 1.0000 all terms. GT dilated 2 vox: blend 0.794, SurfaceDice still 1.000
  (tolerance 2 hides it), topo 0.807, VOI 0.578. Note how much the blend tolerates fattened predictions.
- Speed: ~25 s/patch single-threaded (Betti matching). 200 patches needs a process pool (20 cores here).

## Prior result that reshapes our plan (Jeong, villa #191, #193)
1. Compressed-region failure = **fused sheets, not missed sheets.** At <4 vox spacing (7.91 um => <32 um) models produce one
   probability peak over both sheets in ~78-83% of sites; two resolvable peaks in 0.5%.
2. **No intensity dip between contacting sheets in the CT at 7.91 um**; a model-free phase-demodulation detector sits at chance;
   local geometry features carry no weld info once band thickness is controlled. He calls it a resolution limit.
3. **The official blend does not respond to repairing merges.** VOI term rewards predicted mass. A cross-scroll fine-tune raised
   compressed-bin AUC 0.52 -> 0.89 while the blend fell 0.693 -> 0.624 (topology 0.414 -> 0.301), replicated over 3 seeds.
4. **Two preregistered nulls on label-side interventions**: fine-tuning on repaired weld labels did not change held-out weld
   separation (delta -0.0020, 95% CI [-0.0075, +0.0036], 6 seeds); QC ignore masks moved nothing either.
5. He validated his splitter on a synthetic scroll twin (Diego-dcv/vesuvius-topological-grid) because real welds have no exact GT.

Implications for us:
- "Precise labels" at 7.91 um can't add information that isn't in the CT. Synthetic pretraining that only teaches
  "sheet vs background" at 7.91 um is very likely to hit the same wall.
- A topology-blend "+5%" kill criterion is mis-specified for the merge problem; we need merge-specific endpoints
  (instance VOI_merge per villa PR #1301, component/Betti-0 counts, per-site separation rate as in Jeong's prereg).
- Two angles survive and both favour a simulator:
  a. **Identity through contacts from context** (what unmerge-cli does procedurally): sim is the only source of exact sheet
     identity at welds, for training and for evaluation.
  b. **Resolution**: new scans are 2.4 um / 1.13 um (3.3-7x finer). Sim can render the same geometry at 7.91 / 2.4 / 1.13 um
     with identical labels, i.e. measure where contacts become separable. Needs a real check first: do contacting sheets show
     a gap/dip at 2.4 um in real data?

## Streaming
- `scripts/stream_crop.py` works anonymously from S3: 512³ uint8 at level 0 in ~3 s.
- PHerc1667 2.399 um volume: L0 (37076, 15229, 15229) uint8, chunks 128³, uncompressed, 6 levels. 1.129 um: (42209, 22122, 20276).
- First crop `data/crops/p1667_2p4um_center_z18272` (near mid-height cross-section centroid, not truly dense). Compressed bundles
  show visibly separate strands at 2.4 um; anecdotal only.

## Not done yet
- Baseline surface model inference (HF `scrollprize/surface_recto`, `surface_recto_059_redo`; Jeong's `ckpt_ft_full.pth`).
- Parallel scoring of all 200 patches.
- Truly dense PHerc1667 crops (need umbilicus / dense-region coordinates).
