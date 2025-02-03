# CATHe: Protein Structure Classification with Deep Learning

A CATH protein structure classification model using protein language model embeddings.

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

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your experiment in `config/config.yaml`:
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

3. Train the model:
```bash
python src/train.py
```

You can optionally override any config parameters via command line:
```bash
python src/train.py training.batch_size=64 training.learning_rate=0.0001
```

4. Monitor training:
```bash
tensorboard --logdir logs
```

## License

MIT License - See LICENSE file for details
