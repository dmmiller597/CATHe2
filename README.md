# CATHe: Protein Structure Classifier

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM).

## Project Structure

```
.
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── data/              # Data handling
│   │   └── data_module.py # PyTorch Lightning DataModule
│   ├── models/            # Model definitions
│   │   └── classifier.py  # PyTorch Lightning Module
│   ├── inference.py      # Inference script
│   ├── train.py          # Training script
│   └── utils.py          # Common utilities
├── logs/                  # Training logs and metrics
└── checkpoints/          # Model checkpoints
```

## Installation

### Option 1: Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda create --name cathe python=3.8 -y
conda activate cathe
pip install -r requirements.txt
```

## Configuration

1. Configure your experiment in `config/config.yaml`:
```yaml
data:
  data_dir: "path/to/data"
  train_embeddings: "train_embeddings.npz"
  train_labels: "train_labels.csv"
  # ... other data paths

model:
  embedding_dim: 1024
  hidden_sizes: [512, 256]
  num_classes: 2373
  dropout: 0.2
  use_batch_norm: true

training:
  seed: 42
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
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
```bash
tensorboard --logdir logs
```

## License

MIT License - See LICENSE file for details
