import os
import sys
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils.utils import get_project_root


def load_runs(golden_dir):
    """Load all model checkpoints from golden directory."""
    runs = []
    for folder in sorted(os.listdir(golden_dir)):
        folder_path = os.path.join(golden_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Look for model file
        model_path = os.path.join(folder_path, 'model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(folder_path, 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"Warning: No model found in {folder}")
            continue
        
        checkpoint = torch.load(model_path, map_location='cpu')
        runs.append({
            'folder': folder,
            'config': checkpoint.get('config', {}),
            'history': checkpoint.get('history', {})
        })
    
    return runs

def plot_metrics(runs, golden_dir):
    """Plot validation loss and accuracy for all runs."""
    plt.figure(figsize=(14, 5))
    
    # Plot Validation Loss
    plt.subplot(1, 2, 1)
    for i, run in enumerate(runs):
        history = run['history']
        epochs = history.get('epochs', [])
        val_loss = history.get('val_loss', [])
        if epochs and val_loss:
            plt.plot(epochs, val_loss, label=str(i + 1))
    
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(title='Model #')
    plt.grid(True, alpha=0.3)
    
    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    for i, run in enumerate(runs):
        history = run['history']
        epochs = history.get('epochs', [])
        val_acc = history.get('val_acc', [])
        if epochs and val_acc:
            plt.plot(epochs, val_acc, label=str(i + 1))
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(title='Model #')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(golden_dir, 'comparison_graph.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {save_path}")

def generate_table(runs, golden_dir):
    """Generate comparison table for all runs."""
    headers = [
        'Model #', 'RNN', 'Embedding', 'Emb Dim', 'Bidirectional', 
        'Dropout', 'Layers', 'LR', 'Best Accuracy', 'Best Loss'
    ]
    
    rows = []
    for i, run in enumerate(runs):
        config = run['config']
        history = run['history']
        
        # Get best metrics
        val_acc = history.get('val_acc', [])
        val_loss = history.get('val_loss', [])
        best_acc = max(val_acc) if val_acc else 0
        best_loss = min(val_loss) if val_loss else float('inf')
        
        row = [
            i + 1,
            config.get('rnn_type', '?'),
            config.get('embedding_type', '?'),
            config.get('embedding_dim', '?'),
            'Yes' if config.get('bidirectional', False) else 'No',
            config.get('dropout', '?'),
            config.get('n_layers', '?'),
            config.get('lr', '?'),
            f"{best_acc:.4f}",
            f"{best_loss:.4f}"
        ]
        rows.append(row)
    
    # Print table
    table_str = tabulate(rows, headers=headers, tablefmt='grid')
    print("\n" + "=" * 50)
    print("MODELS COMPARISON")
    print("=" * 50)
    print(table_str)
    
    # Save table to file
    table_path = os.path.join(golden_dir, 'comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write(os.path.basename(golden_dir) + " - MODELS COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(table_str)
    print(f"\nTable saved to: {table_path}")

def main():
    # Get golden_dir from command line or use default
    subdir = sys.argv[1] if len(sys.argv) > 1 else 'experiments'
    golden_dir = os.path.join(get_project_root(),'outputs',subdir)
    
    if not os.path.exists(golden_dir):
        print(f"Error: Directory {golden_dir} does not exist.")
        return
    
    runs = load_runs(golden_dir)
    
    if not runs:
        print("No valid runs found in directory.")
        return
    
    print(f"Found {len(runs)} models in {golden_dir}")
    
    plot_metrics(runs, golden_dir)
    generate_table(runs, golden_dir)

if __name__ == "__main__":
    main()
