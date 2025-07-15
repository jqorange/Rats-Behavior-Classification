# Rats Behavior Classification

This project aims to build a robust multi-modal model for recognising laboratory rat behaviours. Existing approaches often suffer from low temporal resolution, limited label semantics and poor generalisation. Our goal is a 20&nbsp;ms resolution classifier with F1 above 0.9 that can operate across sessions, days and even individual animals. To achieve this we combine video based DeepLabCut (DLC) features with IMU signals and train a semi-supervised model able to predict multiple actions per time step.

## Motivation

Common behaviour classification pipelines exhibit the following shortcomings:

1. Most unsupervised methods such as **B-SOiD** or **keypoint-moseq** only cluster behaviours and the resulting groups are hard to interpret.
2. Fully supervised models (e.g. **DeepEthogram**) require extensive manual labels yet still generalise poorly.
3. They typically predict a single label at each instant while animals often perform multiple actions simultaneously.
4. The temporal granularity is coarse (100–200&nbsp;ms) which misses fast movements.
5. Tasks are simplified to a small set of behaviours leading to over‑fitting.
6. Data are usually captured in constrained environments; models break down in the wild due to noise.
7. Reliance on video alone fails when the subject leaves the frame or images are degraded.

Our system tackles these issues by fusing DLC and IMU features. It uses deep dilated convolutions and a Transformer encoder to model local and long range context. Cross attention merges the two modalities so that when video is unreliable the IMU can compensate.

## Three‑Stage Training Strategy

Training proceeds in three distinct stages:

1. **Stage 1 – Session‑Aware Pretraining**
   * Each session is encoded separately via a session‑aware adapter.
   * Unsupervised contrastive loss learns the internal structure of each session.
2. **Stage 2 – Session Alignment**
   * Encoders are frozen; a small adapter aligns sessions while preserving their structure with a contrastive loss.
   * A supervised contrastive loss attracts samples with overlapping labels across sessions.
3. **Stage 3 – Fine‑Tuning**
   * All parameters are unfrozen.
   * Samples with real labels and high‑confidence pseudo labels (≥0.95 similarity) are trained together with supervised contrastive learning using heavy Gaussian noise.
   * Prototype loss still follows a FixMatch style pseudo‑labelling scheme.

After these stages an MLP classifier is trained. High confidence predictions are added as new training samples to further improve performance.

The full workflow is implemented in `utils/TrainPipline.py`. Checkpoints are saved under `checkpoints/` and can be resumed for continued training.

Use the navigation bar to install the environment, prepare data and follow the quick start guide.