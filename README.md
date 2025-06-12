# Rats Behavior Classification

This project trains a multi-modal model to classify rat behaviors using IMU and DLC data. It combines contrastive pretraining with supervised MLP classification.

## Repository Structure

```
.
├── train.py                # Main script for training
├── plot_data.py            # PCA visualization example
├── checkpoints/            # Directory for trained model checkpoints
├── models/                 # Model architectures and loss functions
└── utils/                  # Data loaders and utility scripts
```

## Data Organization

The dataset must be organized in four directories under a base directory:

- `IMU/` – Raw IMU samples (`.npy` files)
- `DLC/` – Raw DLC samples (`.npy` files)
- `sup_data/` – Supervised training data arrays (`.npy` files)
- `sup_label/` – Labels for supervised training data (`.npy` files)

Paths are defined in `utils/data_loader.py` ([lines 18-22](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/utils/data_loader.py#L18-L22)).

Ensure each session (e.g., `F3D5_outdoor`) has corresponding `.npy` files in these folders. Supervised data arrays combine IMU features (first 35 features) and DLC features (last 20 features) ([lines 75-84](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/utils/data_loader.py#L75-L84)).

## Model Architecture

- **EncoderFusion**: Processes IMU and DLC inputs separately, then fuses features using cross-attention and gating ([fusion.py lines 13-63](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/models/fusion.py#L13-L63)).
- **Encoder**: Uses dilated convolutions and transformer layers to capture temporal context ([encoder.py lines 7-55](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/models/encoder.py#L7-L55)).
- **MLPClassifier**: Simple two-layer MLP for behavior classification ([classifier.py lines 6-21](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/models/classifier.py#L6-L21)).
- **Dynamic Weight Averaging**: Adjusts loss weights dynamically during training ([dwa.py lines 10-23](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/models/dwa.py#L10-L23)).

Loss functions (e.g., hierarchical contrastive loss, supervised contrastive loss) are implemented in `models/losses.py`.

## Training Pipeline

The training process involves the following steps, defined in `utils/TrainPipline.py` ([lines 314-344](https://github.com/jqorange/Rats_Behavior_Classification/blob/main/utils/TrainPipline.py#L314-L344)):

- Load and align session data
- Initialize model components
- Test individual components
- Contrastive learning pretraining
- Supervised MLP training
- Evaluate performance on test data

### Training Configuration

Training parameters can be modified in `train.py`:

```python
trainer_params = {
    'mask_type': 'binomial',
    'd_model': 128,
    'nhead': 4,
    'hidden_dim': 4,
    'lr_encoder': 0.0001,
    'lr_classifier': 0.001,
    'batch_size': 128,
    'contrastive_epochs': 1,
    'mlp_epochs': 5,
    'save_path': save_path,
    'save_gap': 5,
    'n_cycles': 100,
    'n_stable': 40
}
```

## Usage

1. Place your dataset under the directory specified by `data_path` in `train.py` (default example: `D:\Jiaqi\TrainData`).

2. Adjust the `session_name` list in `train.py` to match your data sessions.

3. Run training:

```bash
python train.py
```

The model checkpoints are saved in the `checkpoints/` folder.

### Example Inference

The inference example is provided at the end of `train.py`. Modify paths and sessions accordingly.

### Visualization

Use `plot_data.py` to visualize data distributions using PCA in 3D.

## License

This project is licensed under the MIT License.

---



