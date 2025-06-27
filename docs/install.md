# Installation

The code requires Python 3.10 and PyTorch with CUDA support. The easiest way is to use conda:

```bash
conda create -n Behavior python=3.10
conda activate Behavior
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit=11.8
pip install plotly
```

After installation you can run the training scripts directly from the project root.