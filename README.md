# CATHe

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM).

CATHe is built using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training, [Hydra](https://hydra.cc/docs/intro/) for simple configuration, and [Weights & Biases](https://docs.wandb.ai/quickstart) for tracking experiments and results.

## Project Structure

```
.
├── config/
│   └── config.yaml         # Configuration file
├── scripts/
│   └── download_data.sh    # Download and organize data
├── src/
│   ├── data/               # Data handling
│   │   └── data_module.py  # PyTorch Lightning DataModule
│   ├── models/             # Model definitions
│   │   └── classifier.py   # PyTorch Lightning Module
│   ├── inference.py        # Inference script
│   ├── train.py            # Training script
│   └── utils.py            # Common utilities
├── logs/                   # Training logs and metrics
├── checkpoints/            # Model checkpoints
└── requirements.txt        # Project dependencies
```

## Data

The dataset used for training, optimizing, and testing CATHe was derived from the CATH database. The datasets, along with the weights for the CATHe artificial neural network, can be downloaded from Zenodo from this link: [Dataset](https://doi.org/10.5281/zenodo.6327572).

To download the CATHe dataset from Zenodo, run the following script from your terminal:

```bash
bash scripts/download_data.sh
```

## Installation

### Option 1: Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda create --name cathe python=3.8 -y
conda activate cathe
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to set paths and parameters:
```yaml
data:
  data_dir: "data"
  ...

model:
  embedding_dim: 1024
  hidden_sizes: [512, 256]
  ...

```

## Usage

### Train the model:
```bash
python src/train.py
```

You can optionally override any config parameters via command line:
```bash
python src/train.py training.batch_size=64 training.learning_rate=0.0001
```

### Monitor training:
After starting training, visit your Weights & Biases dashboard at https://wandb.ai.
If you haven't already, run:
```bash
wandb login
```
to set up your W&B credentials.

## License

MIT License - See LICENSE file for details

