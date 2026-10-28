# 2026-10-16: public baseline (bshepp/vesuvius-nnunet-model) on jeong200

## Model
- Hugging Face `bshepp/vesuvius-nnunet-model`, MIT licence.
- nnU-Net v2 PlainConvUNet, 31M parameters, 7.91 um, trained on all 786 Kaggle training volumes. Kaggle private LB 0.449.
- **Contaminated for jeong200**: the pool comes from the Kaggle training set. Use this model only to characterise
  failure modes, never as a held-out number.
- Loading: `eval/infer_nnunet.py` uses torch `weights_only=True` with numpy scalars allowlisted, then
  `nnUNetPredictor.manual_initialization`. nnU-Net's unrestricted `torch.load` on the untrusted pickle never runs.
- Predictions in `data/preds/bshepp/`.

## Label 2 pitfall
- The model's class 2 is the Kaggle "unlabeled" (ignore) label, and the model predicts it, sometimes for 98% of a patch.
- Plain argmax therefore drops surfaces. Surface is decided as p(class 1) > p(class 0) instead.
- Argmax predictions are kept in `preds/bshepp_argmax_cls1` for reference.

## Results (200 patches; tolerance 2 vox)
| scroll | n | precision@2 | recall@2 (median) | recall p10 | GT fg | pred fg |
|---|---|---|---|---|---|---|
| s1 | 162 | 0.81 | 0.43 | 0.10 | 0.174 | 0.072 |
| s4 | 36 | 0.82 | 0.49 | 0.26 | 0.107 | 0.062 |
| s5 | 2 | 0.92 | 0.72 | 0.66 | 0.141 | 0.097 |

- Correlation(GT foreground fraction, recall) = -0.69. In dense / compressed patches the model **misses** sheets,
  rather than fusing them.
- Separation endpoint (`eval/separation.py`, `data/preds/sep_bshepp.json`):
  - Compressed sites: hit 1.0%, sep_cc 0.8%, sep_cc given hit 85%.
  - Wide sites: hit 25.6%, sep_cc 25.6%.
  - With hits this rare, the conditional number is not meaningful.
- Implication: one plausible failure mode in hard regions is "dense = unlabeled". Kaggle ignore regions are likely
  concentrated where annotators gave up, so models learn to abstain there. Synthetic data with exact labels everywhere
  (no ignore regions) attacks this directly. Report recall at compressed sites next to separation.

## Separation endpoint sanity (12 sanity patches)
| predictions | compressed sep_cc | wide sep_cc |
|---|---|---|
| identity | 99.8% | 100% |
| dilate2 | 99.8% | 100% |
| dilate3 | 16% (line gap still 99.8%) | 99.9% |
| dilate4 | 0% | 95% |

The local-component test is stricter than the line-gap test.
