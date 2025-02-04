# CATHe

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM).

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

## Download Data

To download the CATHe dataset from Zenodo, run the following script from your terminal:

```bash
bash scripts/download_data.sh
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
  data_dir: "data"
  train_embeddings: "embeddings/SF_Train_ProtT5.npz"
  val_embeddings: "embeddings/SF_Val_ProtT5.npz"
  test_embeddings: "embeddings/SF_Test_ProtT5.npz"
  train_labels: "annotations/Y_Train_SF.csv"
  val_labels: "annotations/Y_Val_SF.csv"
  test_labels: "annotations/Y_Test_SF.csv"

model:
  embedding_dim: 1024
  hidden_sizes: [128, 128, 128]
  dropout: 0.5
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler_factor: 0.1
  scheduler_patience: 10

training:
  seed: 42
  batch_size: 256
  max_epochs: 200
  early_stopping_patience: 30
  num_workers: 4
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: "16-mixed"
  log_every_n_steps: 50
  save_top_k: 3
  monitor_metric: "val_acc"
  monitor_mode: "max"
  output_dir: "outputs"
  checkpoint_dir: "${training.output_dir}/checkpoints"
  log_dir: "${training.output_dir}/logs"
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

### Run Inference
Generate predictions using the inference script:
```bash
python src/inference.py --checkpoint path/to/model.ckpt --embeddings path/to/embeddings.npz --output predictions.csv
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
