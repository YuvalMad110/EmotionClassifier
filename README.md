# Emotion Classifier

An RNN-based text classification model for detecting emotions in tweets. Supports LSTM and GRU architectures with pre-trained embeddings (GloVe, Word2Vec) or trainable embeddings.

**Author:** Yuval Mad

## Features

- **Flexible RNN Architecture**: Choose between LSTM and GRU cells
- **Pre-trained Embeddings**: Support for GloVe (50/100/200/300d) and Word2Vec (300d)
- **Configurable Hyperparameters**: All parameters adjustable via command line
- **Training Utilities**: Automatic logging, metrics visualization, and model checkpointing
- **Batch Experiments**: Run multiple experiments with different configurations
- **Results Analysis**: Compare and visualize results across experiments

## Project Structure

```
EmotionClassifier/
├── data/
│   ├── embeddings/              # Pre-trained embedding files
│   ├── dataset_ec.py            # Dataset and DataLoader classes
│   ├── embeddings_ec.py         # Embedding loading utilities
│   ├── vocabulary_ec.py         # Vocabulary building and tokenization
│   ├── download_glove.py        # Script to download GloVe embeddings
│   └── download_word2vec.py     # Script to download Word2Vec embeddings
├── outputs/
│   └── experiments/             # Training outputs directory
├── utils/
│   └── utils.py                 # Utility functions
├── ec_main.py                   # Main training script
├── ec_model.py                  # Model architecture
├── ec_trainer.py                # Training loop and logging
├── main_wrapper.py              # Batch experiments runner
├── analyze_outputs.py           # Results analysis and comparison
└── README.md
```

## Usage

### Single Training Run

```bash
python ec_main.py --embedding_type glove --embedding_dim 100 --rnn_type lstm --epochs 50
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_path` | - | Path to training CSV |
| `--val_path` | - | Path to validation CSV |
| `--test_path` | None | Path to test CSV (optional) |
| `--embedding_type` | glove | Embedding type: `glove`, `word2vec`, `trainable` |
| `--embedding_dim` | 100 | Embedding dimension: 50, 100, 200, 300 |
| `--embedding_trainable` | False | Fine-tune embeddings during training |
| `--rnn_type` | gru | RNN cell type: `lstm`, `gru` |
| `--hidden_dim` | 256 | Hidden layer dimension |
| `--n_layers` | 2 | Number of RNN layers |
| `--bidirectional` | False | Use bidirectional RNN |
| `--dropout` | 0.2 | Dropout probability |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 30 | Number of training epochs |
| `--outputs_subdir` | experiments | Output subdirectory name |

### Batch Experiments

Edit `main_wrapper.py` to define multiple experiment configurations:

```python
OUTPUTS_SUBDIR = 'my_experiments'

EXPERIMENTS = [
    {'lr': 0.001, 'dropout': 0.2, 'epochs': 50},
    {'lr': 0.001, 'dropout': 0.3, 'bidirectional': True},
    {'embedding_type': 'word2vec', 'embedding_dim': 300},
]
```

Run all experiments:

```bash
python main_wrapper.py
```

### Analyze Results

```bash
python analyze_outputs.py experiments  # Analyze outputs/experiments/
python analyze_outputs.py lstm         # Analyze outputs/lstm/
```

This generates:
- `comparison.png`: Validation loss and accuracy plots for all models
- `comparison.txt`: Summary table with hyperparameters and best metrics

## Dataset

The model expects CSV files with `text` and `label` columns. Labels correspond to 6 emotion classes:

| Label | Emotion |
|-------|---------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

## Results

Best model achieves **94.88% validation accuracy** using LSTM with GloVe 300d embeddings.

| Model | Embedding | Dim | Bidirectional | Dropout | Layers | Accuracy |
|-------|-----------|-----|---------------|---------|--------|----------|
| LSTM | GloVe | 300 | No | 0.2 | 2 | 94.88% |
| LSTM | Word2Vec | 300 | No | 0.2 | 2 | 94.73% |
| GRU | Word2Vec | 300 | No | 0.2 | 2 | 94.52% |
