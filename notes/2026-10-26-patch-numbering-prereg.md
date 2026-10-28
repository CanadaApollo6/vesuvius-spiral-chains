# Prereg: our chains assign windings to Stevens' Scroll 4 patches (vs Kelly's index), 2026-10-26

**Frozen before any label metric was computed.** Already looked at:
- patch counts per band from Kelly's centroids;
- the affine check, which compared 10 sample patches against our m7 mask, not against labels;
- the patch-id sets.

No human label has been read against a patch or a chain-patch hit. The scoring code is written after this note; its
sha256 goes into the results section with any deviation logged. Scout: notes/2026-10-26-patch-numbering-scout.md.

## Data (all fixed)

- **Patches.** Stevens' `s4_good_patches.zip` (47,797) and `s4_bad_patches.zip` (9,171) from
  dl.ash2txt.org/community-uploads/will/. sha256 values `5c5a4865…3fb2` and `72f436c5…bf6b` match Kelly's table. They
  are read from the zips in memory with `python -I`; nothing is executed.
  - Frame: 20231117161658 L0 (7.91 µm).
  - Mapping: `data/scratch/patch_scout/affine_s4_7p91_to_2p399.json` (public metadata.json volume_transform) then /4 →
    20251217075048 L2 (9.596 µm), our frame.
  - Patch id = N in `patch_N`. The two sets are disjoint.
- **Bands (L2 z).** B1 = [6500, 8000), the pilot band. B2 = [4500, 6000).
  - A patch belongs to a band if its **medoid** (the valid vertex nearest the mean of its valid vertices, after mapping)
    has z in the band.
- **Chains.**
  - B1: `data/scratch/auto_windings/p1667/band_m7_p1667_ray_c/chains.json` (the confirmed M2 recipe, `--raise off`).
  - B2: `data/scratch/ink_lowband/autowind_z4500/chains.json` (autowind CLI, `--raise off`).
  - **Primary:** chains split at steps with p_err ≥ 0.5 (bridged steps have p_err 1.0), segments with at least 2
    points. **Secondary:** unsplit.
- **Kelly.** `community/wrap_index.csv` (k_q3c, component, thN radians, cut at thN = 0) and `solve_edges.json` (joins:
  a, b, msep_um, verdict, d_i). This is their published whole-scroll solve, used as is.
- **Human labels.** `data/eval/p1667/labels_merged_v4.npz` (k per merged-mesh vertex, centre, seam 144°, sign −1, K0 39)
  on `data/dl_p1667/merged_v4_2p399` (L2 = 2.399 / 4).

## Definitions

- **Hit.** A chain point hits a patch when some vertex of that in-band patch is within **3.0 L2 vox**. All patch
  vertices are used, and one point may hit several patches.
- **Chain terms.** Within one chain segment, collect the (patch, wind) hits.
  - A patch hit at more than one wind in a segment is a **switch observation**. It gives no terms from that segment.
  - The remaining hit patches are sorted by wind, ties by patch id. Each consecutive pair (a, b) gives the term
    d = wind_b − wind_a, which is 0 for patches hit at the same point or wind.
  - A (chain, a, b) triple counts once.
  - **Chain-tied pairs** are the distinct (a, b) pairs with at least one term.
- **Continuous turn coordinate T** (any winding index with a seam):
  - T = k + s·φ/360, where φ is the angle past the index's seam (degrees, mod 360) and s is the change in k when the seam
    is crossed in +θ.
  - **Seam-free difference** D(a, b) = round(T_b − T_a − s·Δθ_short(a→b)/360), where Δθ_short is the short-arc angle
    from a to b in that index's own θ frame.
  - Human: θ is the degrees of atan2(y − c_y, x − c_x) around the label `centre`, φ = (θ − 144) mod 360, s = −1, so
    T_H = K0 − (Phi − 144)/360, which is continuous.
  - Kelly: φ = thN·180/π, s = s_K.
  - Ours: θ is taken around `umbilicus_p1667_L2.json`, seam at θ = 0, s = s_O.
- **Patch reference point** = the medoid.
  - Human T(p) = median over matched vertices of T_H,v − s_H·Δθ_short(v→medoid)/360.
  - Our angle for a patch = the medoid angle.
  - Kelly's angle for a patch = their thN.
- **Human label of a patch.**
  - Every second row and column of patch vertices is matched to the nearest valid merged-mesh vertex within
    **4.0 L2 vox**.
  - The patch is **labelled** if at least 30 vertices and at least 30 % of the queried vertices match, and fewer than
    10 % of the matched vertices have |T'_v − median| ≥ 0.5.
  - If the first two conditions hold but the last fails, it is **human-ambiguous**.
  - Otherwise it is unlabelled.
- **Label-free orientation choices** (made before scoring):
  - s_K ∈ ±1: the one that maximises the share of geometric same-wrap joins (msep_um ≤ 0.59·134 = 79.06) with
    D_K = 0.
  - σ_K ∈ ±1 (Kelly vs outward): the one that maximises agreement of σ_K·D_K with chain d on chain-tied pairs.
  - s_O: the lower LP objective of the two runs.
  - Human outward is +; human k rises outward (11 → 41).

## Methods (integer solves are LP on difference terms, not spiral fits)

LP: minimise Σ w·|k_b − k_a − c| with HiGHS. The constraint matrix is totally unimodular, so the vertex solution is
integral; it is rounded anyway.
- Nodes are in-band patches present in at least one term.
- Components come from the terms. Components with fewer than 20 patches count as unnumbered.
- c = D_target + s_O·cross_O(a, b), where cross_O = ±1 if the short arc crosses θ = 0, else 0.

| Method | Terms |
|---|---|
| (a) Kelly | none: the published k_q3c with the seam (s_K, thN) |
| (b) chains only | chain terms (D_target = d, w = 1) + geometric same-wrap overlap terms (Kelly joins with both ends in band and msep_um ≤ 79.06 µm: D_target = 0, w = 1) |
| (c) Kelly + chains | Kelly joins with both ends in band (D_target = σ_K·D_K(a, b) from their solved index, w = 1) + chain terms (w = 1) |

## Metrics

- **Gate (label-free, B1 only).** Coverage = the share of in-band patches (good + bad) hit by at least one chain point
  (primary chains). **Continue only if coverage ≥ 30 %.** Otherwise stop and report.
- **M1, pairwise on chain-tied pairs** (both patches labelled; primary metric for the chain evidence). This is the share
  of pairs where the method's D equals D_H:
  - chains: d, where a pair's d is the majority over its terms, with ties counted wrong;
  - Kelly: σ_K·D_K;
  - (b) and (c): D from their solution, when both patches are numbered.
- **M2, global, primary.** For each method:
  - R_p = T_H(p) − T_m(p), with T_m oriented outward.
  - Per component, the offset o maximises #(|R_p − o| < 0.5). A patch is correct if |R_p − o| < 0.5.
  - The value reported is the **share of labelled in-band patches correct**. Unnumbered patches count as wrong.
  - Also reported: coverage, accuracy among numbered patches, and a vertex-weighted version.
  - σ ∈ ±1 is also fitted as a check; a disagreement with the label-free orientation is reported.
- **M3, bad-patch detection.** These patches are eligible: in band and hit at 2 or more chain points (unsplit chains).
  - Chain flag = at least one switch observation.
  - Baseline flag = the patch is in at least one of Kelly's unsatisfied joins (`community/unsatisfied.csv`).
  - Precision and recall are reported for each flag against (i) Stevens' bad set and (ii) human-ambiguous patches.
- **M4, disputes.**
  - A disputed pair is a chain-tied pair where the majority chain d ≠ σ_K·D_K.
  - Ranking: the number of distinct chains asserting that d (descending), then the smallest per-chain max p_err between
    the two hits (ascending).
  - Written to CSV.
  - Scored: among all disputes and the top 50 and top 200 (both patches labelled), the share where human agrees with the
    chains, with Kelly, or with neither.

## Decision rules

- **"Chains improve Kelly's index"** requires that M2(c) − M2(a) > 0 with the 95 % cluster-bootstrap lower bound > 0 in
  **both** bands.
  - Clusters are patches grouped by (250-vox z bin × 30° sector), 2,000 resamples, paired.
  - If only one band passes, the result is "mixed" and is reported as such.
- **"Chains beat Kelly pairwise"** requires that M1(chains) − M1(Kelly) on the same pairs > 0, with a bootstrap over
  chains (rays) and lower bound > 0, in both bands.
- **M3 is descriptive.** There is no claim threshold.
- **No tuning.** Weights, tolerances, the split threshold and the band set are as stated. Any extra variant is labelled
  exploratory.

## Caveats, stated in advance

- The human labels are one merged read (81 % single-wrap agreement at build).
- Our PHerc1667 umbilicus came from that mesh. B1 is our confirmation band and B2 our ink band, so both have been seen.
- Kelly's index was built without these labels (their reference anchors were used only in their fit evaluation).
- (c) uses Kelly's solved differences, not their raw measurements.
- The patch geometry and join list come from Stevens and Kelly. Only the chain evidence is ours.

## Results (appended 2026-10-26 after the run; text above this heading unchanged since freezing)

The frozen prereg sha256 (before this section) is `e0ae7064…4382`, recorded in
`data/scratch/patch_scout/prereg_sha256_frozen.txt`.

Scripts are in `data/scratch/patch_numbering/`:
- `extract.py` sha `67fcae1f…3e0b`;
- `score.py` sha `b2c513e5…6f26`.

Outputs are `B{1,2}_results.json`, `B{1,2}_disputes.csv` and `B{1,2}.log`. Everything ran on CPU in about 2 minutes per
band.

**Deviations, all made before any label metric was printed:**
1. **Sign typo in the prereg.** The "Human T(p)" line says `T_H,v − s_H·Δθ(v→medoid)/360`. The transport consistent
   with the frozen D definition and with T continuity is `T_H,v + s_H·Δθ(v→medoid)/360`, and that is what the code
   uses.
2. **Crash fix.** A crash on a patch missing from Kelly's index (one of their 34 unplaced patches) was fixed, and both
   bands were rerun. Only the label counts had printed before the crash.
3. **M4 top-50 / top-200.** These are taken in rank order among disputes where both patches are labelled.

### Setup outcomes (label-free)

- **Gate.** B1 coverage is 73.3 % (B2: 78.0 %), against the 30 % threshold, so the gate passes.
- **Terms.** 29.3k primary chain terms per band, spread over 19-20k chain-tied pairs.
- **Orientation choices.**
  - s_K = +1: same-wrap joins have D_K = 0 at 98.7 % (s_K = −1 gives 96.4 %).
  - σ_K = +1: 17,910 vs 9,855 chain agreements.
  - s_O = −1: LP objective about half that of +1.
  - The σ check with negated T scores about 3 %, so the label-free orientations are right.
- **Human labels.**
  - B1: 9,472 patches labelled, 1,104 ambiguous, 1,684 unlabelled.
  - B2: 9,803 labelled, 675 ambiguous, 1,494 unlabelled.

### Results table

| Metric | B1 z 6500-8000 | B2 z 4500-6000 |
|---|---|---|
| **M2** share of labelled patches with the right global winding (per-component offset): (a) Kelly | 97.68 % | 98.61 % |
| (b) chains + geometric overlaps (no Kelly measurements) | **98.34 %** (+0.67 pp, CI [+0.14, +1.29]) | **99.18 %** (+0.57 pp, CI [−0.08, +1.19]) |
| (c) Kelly + chains | 97.86 % (+0.18 pp, CI [−0.02, +0.37]) | **99.44 %** (+0.83 pp, CI [+0.29, +1.35]) |
| (b) / (c) with unsplit chains | 98.16 % / 97.86 % | 99.31 % / 99.41 % |
| Vertex-weighted M2: a / b / c | 98.65 / 99.02 / 98.73 % | 99.12 / 99.54 / 99.64 % |
| **M1** pairwise on chain-tied labelled pairs (n): chain majority d | 96.05 % (14,225) | 97.47 % (15,504) |
| Kelly | 99.46 % | 99.61 % |
| (b) / (c) | 99.76 / 99.61 % | 99.85 / 99.85 % |
| Chains − Kelly (CI by chain) | −3.41 pp [−3.81, −3.03] | −2.15 pp [−2.44, −1.87] |
| M1, pairs with D_H ≠ 0 only: chains / Kelly | 91.8 / 99.2 % | 94.6 / 99.4 % |
| **M3** bad-patch detection (eligible patches, bad base rate): chain switch flag P / R | 2/13 flagged, R 0.2 % (6,484; 14.8 %) | 0/6 flagged, R 0 % (7,148; 16.0 %) |
| Kelly unsatisfied-join flag P / R | 32.1 % / 53.8 % | 32.7 % / 50.0 % |
| vs human-ambiguous: chain flag / Kelly flag recall | 0 % / 62.6 % (base 4.2 %) | 0 % / 67.8 % (base 3.0 %) |
| **M4** disputes (chain majority ≠ Kelly): count / with both patches labelled | 1,255 / 582 | 1,076 / 420 |
| Human sides with chains / Kelly / neither: all | 10.1 / 89.5 / 0.3 % | 12.9 / 86.9 / 0.2 % |
| Top 50 | 6 / 94 / 0 % | 26 / 74 / 0 % |
| Top 200 | 15.5 / 84.5 / 0 % | 21 / 79 / 0 % |

CIs are 95 %, paired, 2,000 resamples. M2 CIs are clustered by 250-vox z bin × 30° sector; M1 CIs by chain.

### Verdict under the frozen rules

- **"Chains improve Kelly's index" (M2(c) − M2(a) > 0 with lower bound > 0 in both bands): MIXED.** B2 passes at
  +0.83 pp. B1 does not: +0.18 pp, lower bound −0.02.
- **"Chains beat Kelly pairwise": NO.** Chains are 2-3.4 pp worse in both bands. On the pairs where they disagree, the
  human sides with Kelly about 87-90 % of the time.
  - Chain pairwise accuracy (96.1 / 97.5 %) matches our earlier PHerc1667 step precision (95.8 %). That is a sanity check
    on the hit and label machinery.
- **M3 (descriptive).** Our chains are useless as a sheet-switch detector here: a switch flag almost never fires, 19
  flags across 13.6k eligible patches. Kelly's unsatisfied-join flag has about 2x lift.
- **M4.** The ranking does not concentrate correct chain calls, so it is not a useful human-check queue in its current
  form.
- **Noted, not claimed.** (b) uses only the chains plus Stevens' patch overlaps (Kelly's geometric same-wrap rule, none
  of Kelly's radial or CT measurements). It beats Kelly's published index by point estimate in both bands, and its CI
  excludes 0 in B1.
  - The decision rule was frozen for (c) only, so this is a lead, not a result.
  - It could partly be a band-local-solve effect versus Kelly's whole-scroll solve. Kelly's whole-scroll terms were not
    re-solved band-locally, because (c)'s Kelly terms are their solved differences, so a Kelly-only band LP reproduces (a).
- **Context.** Kelly's index is already about 98-99 % right on these bands. That leaves about 130-220 wrong patches per
  band, so the room for any helper is about 1-2 pp.
