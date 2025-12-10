import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

from data.dataset_ec import get_data
from ec_model import EmotionClassifier
from ec_trainer import Trainer
from utils.utils import get_project_root

# Full run command example:
# CUDA_VISIBLE_DEVICES=1 python3 ./Projects/EmotionClassifier/ec_main.py --batch_size 128 --epochs 100

def parse_arguments():
    """
    Parses command line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train an RNN model for Emotion Detection')

    # --- Data Arguments ---
    parser.add_argument('--train_path', type=str, default='/home/yuvalmad/datasets/tweets/train.csv', help='Path to training CSV')
    parser.add_argument('--val_path', type=str, default='/home/yuvalmad/datasets/tweets/validation.csv', help='Path to validation CSV')
    parser.add_argument('--test_path', type=str, default=None, help='Path to test CSV (optional)')
    parser.add_argument('--embedding_dir', type=str, default='data/embeddings', help='Directory for embedding files - relative to project root')
    
    # --- Embedding Arguments ---
    parser.add_argument('--embedding_type', type=str, default='glove', choices=['glove', 'word2vec', 'trainable'], help='Type of embeddings to use')
    parser.add_argument('--embedding_dim', type=int, default=100, choices=[50, 100, 200, 300], help='Dimension of embeddings (ignored for word2vec)')
    parser.add_argument('--embedding_trainable', action='store_true', help='If set, embeddings will be fine-tuned during training')
    
    # --- Model Architecture Arguments ---
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='Type of RNN cell')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--bidirectional', action='store_true', help='If set, uses Bidirectional RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Max sequence length for truncation (None = no truncation)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--val_frequency', type=int, default=1, help='Run validation every X epochs')
    
    return parser.parse_args()

def main():
    # 1. Parse Arguments
    args = parse_arguments()
    config = vars(args)
    
    print(f"\n{'='*10} Configuration {'='*10}")
    for k, v in config.items():
        print(f"{k}: {v}")
    print('='*35)

    # 2. Prepare Data
    print("\n[1/4] Preparing Data...")
    embedding_dir_full = os.path.join(get_project_root(), args.embedding_dir)
    data = get_data(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        embedding_trainable=args.embedding_trainable,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        embedding_dir=embedding_dir_full
    )
    
    # Update embedding_dim in config (important for word2vec which has fixed dim)
    config['embedding_dim'] = data['embedding_dim']
    config['num_classes'] = data['num_classes']
    config['oov_len'] = len(data['oov_words'])
    config['oov_words'] = data['oov_words']
    # print('test oov words: (False = in vocab, True = OOV)')
    # for word in ['t', 'm', 've']:
    #     print(f'{word}: {word in oov_words}')

    # 3. Initialize Model
    print("\n[2/4] Initializing Model...")
    model = EmotionClassifier(
        embedding_layer=data['embedding'],
        hidden_dim=args.hidden_dim,
        output_dim=data['num_classes'],
        n_layers=args.n_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        rnn_type=args.rnn_type
    )
    
    # Calculate and print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> Model Architecture: {args.rnn_type.upper()} (Bi-directional: {args.bidirectional})")
    print(f"   -> Embedding: {args.embedding_type} (dim={data['embedding_dim']}, trainable={args.embedding_trainable})")
    print(f"   -> Vocab size: {data['vocab_size']}")
    print(f"   -> Trainable Parameters: {trainable_params:,}")
    
    # Print OOV info if available
    if data['oov_words']:
        print(f"   -> OOV words: {len(data['oov_words'])}")

    # 4. Setup Training Components (Optimizer & Loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        config=config
    )

    # 5. Run Training
    print("\n[3/4] Starting Training...")
    trainer.fit(
        num_epochs=args.epochs,
        val_frequency=args.val_frequency
    )
    
    # Optional: Test Evaluation (only if test set was provided)
    if data['test_loader']:
        print("\n[4/4] Running Evaluation on Test Set...")
        test_loss, test_acc = trainer._run_epoch(data['test_loader'], is_train=False)
        print(f"   -> Test Results: Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
        trainer.logger.info(f"Test Results | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    print(f"\nDone! Results saved in: {trainer.outputs_dir}")

if __name__ == "__main__":
    main()
