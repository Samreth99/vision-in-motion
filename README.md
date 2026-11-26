**Is Per-Frame Semantics Enough for Motion Understanding?**

[DINOv3](https://arxiv.org/abs/2508.10104) (image-only) is state of the art on many visual benchmarks but falls short on
[Something-Something v2](https://arxiv.org/abs/1706.04261) (SSv2), a motion-understanding dataset. [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) (video + image)
reaches 77.7% on SSv2. With **frozen encoders**, can we close the gap by increasing the head's
parameters or by using a temporal architecture over per-frame features? On which classes
does the gain appear?

## Setup

- **Data:** [SSv2](https://arxiv.org/abs/1706.04261) (174 classes; 168,913 train / 24,777 val videos), possibly a subset.
- **Input:** 16 uniformly sampled frames (same indices for both models), center crop resized
  to 256×256, ImageNet normalization.
- **Encoders (frozen):** [DINOv3](https://arxiv.org/abs/2508.10104) ViT-B/16 and [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) ViT-B/16, official checkpoints.

| ID  | Frozen encoder input                          |
| --- | --------------------------------------------- |
| (a) | V-JEPA 2.1, joint encoding of a 16-frame clip |
| (b) | V-JEPA 2.1, independent frames                |
| (c) | DINOv3, independent frames                    |

## Experiments

- **E1 — Linear probes:** one linear layer on mean-pooled (patches and time) features of (a), (b), (c).
- **E2 — Capacity vs. architecture:** five heads on the per-frame features of (b) and (c),
  accuracy plotted against parameter count, with (a) as reference:
  1. Mean + Linear (E1 baseline)
  2. Mean + Linear + GELU + Linear (more capacity, order-blind)
  3. Concat + Linear (keeps frame order)
  4. Temporal Conv + GELU + Mean + Linear (keeps frame order)
  5. Project + Time tag + Self-attn + FFN + Mean + Linear (keeps frame order)

## Evaluation

- Top-1 accuracy on the validation split, 3 seeds (mean ± std), paired-bootstrap CIs for key differences.
- Hyperparameters tuned on a held-out 10% of the training subset.
- Shuffle test: each head evaluated with original vs. shuffled frame order.
- Direction-sensitive classes (e.g. pushing left-to-right vs. right-to-left), chosen from class names in advance.
- Sanity check: a small subset at 384 px to verify that using 256 px does not hurt V-JEPA.

## Repository layout

```
configs/     data.yaml, e1_linear.yaml, e2_heads/ (one yaml per head)
src/
  data/      SSv2 loader, frame sampling, subset split, 10% held-out
  encoders/  dinov3.py, vjepa2.py (wrap official checkpoints)
  features/  extract and cache features
  heads/     the five head modules
  train.py
  eval/      accuracy.py, shuffle_test.py, direction_classes.py
scripts/     run_e1.sh, run_e2.sh, make_plots.py
data/        gitignored: raw SSv2, cached features
results/     json/csv per run
figures/     accuracy-vs-parameters plot, per-class gains
notebooks/   analysis only
report/      LaTeX report + slides
```
