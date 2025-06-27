# Usage Guide

All training and evaluation logic is wrapped by `utils/TrainPipline`. The entry point `train.py` creates a `TrainPipline` instance and runs the full three‑stage curriculum described on the home page.

Most hyper parameters such as model dimensions and learning rates can be edited in `train.py` or passed to `FusionTrainer` through the pipeline.

### Additional Scripts

* **inference.py** – generate representations from saved checkpoints.
* **plot_embeddings.py** – draw PCA plots for visual inspection.
* **evaluate.py** – evaluate predictions on CSV labels.
* **multi_class_classifier.py** / **train_repr_mlp.py** – self training utilities for representation level classifiers.

Consult [Files](usage/file.md) for a file‑by‑file description and [Functions](usage/function.md) for API level documentation.