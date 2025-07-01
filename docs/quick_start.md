# Quick Start

1. Install dependencies following [Installation](install.md).
2. Prepare the dataset as described in [Data Structure](DataStructure.md).
3. Train the encoders and classifier:

```bash
python train.py
```

Checkpoints are written to `checkpoints/` every few epochs.

4. Generate latent representations for specific sessions:

```bash
python inference.py --data_path <path_to_dataset> --sessions F3D5_outdoor F3D6_outdoor
```

5. Visualise the embeddings using PCA:

```bash
python plot_embeddings.py --rep_dir representations --sessions F3D5_outdoor
```

6. Evaluate predictions on the labelled segments:

```bash
python evaluate.py --pred_dir predictions --label_path <labels>
```