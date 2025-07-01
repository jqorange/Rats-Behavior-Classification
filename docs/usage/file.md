# File Overview

## Top level scripts

* **train.py** – launches the full training pipeline defined in `utils/TrainPipline`.
* **inference.py** – loads checkpoints and produces latent representations for selected sessions.
* **plot_embeddings.py** – visualise representations with PCA.
* **evaluate.py** – compute metrics for prediction CSVs on labelled frames.
* **multi_class_classifier.py** / **train_repr_mlp.py** – utilities for self training a representation‑level MLP.
* **multi_class_inference.py** – run the saved MLP and output probabilities.
* **predict_repr_mlp.py** – inference using a MLP trained on representations.

## Model code (`models/`)

* **dilated_conv.py** – residual blocks of dilated 1D convolutions.
* **encoder.py** – single‑modal encoder combining domain adapters, dilated conv blocks and a Transformer.
* **fusion.py** – cross‑attention based fusion of IMU and DLC encoders.
* **classifier.py** – simple MLP classifier for sequence level labels.
* **deep_mlp.py** – deeper MLP used for representation level self training.
* **domain_adapter.py** – small modules inserted before encoders to align or tag sessions.
* **masking.py** – utilities to generate random temporal masks.
* **losses.py** – contrastive and prototype loss implementations.

## Utilities (`utils/`)

* **data_loader.py** – loads IMU/DLC numpy files and their supervised counterparts.
* **trainer.py** – core optimisation logic with contrastive and MLP phases.
* **TrainPipline.py** – high level wrapper orchestrating data loading, model setup and evaluation.
* **tools.py** – helper functions for sequence slicing.