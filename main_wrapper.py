import subprocess
import sys

# =============================================================================
# EXPERIMENTS CONFIGURATION
# =============================================================================
# Output subdirectory - all experiments will be saved under ./outputs/{OUTPUTS_SUBDIR}/
OUTPUTS_SUBDIR = 'gru'

# Edit this list to define the experiments.
# Each dict contains only the parameters you want to override from defaults.
# Available parameters:
#   --train_path, --val_path, --test_path, --embedding_dir
#   --embedding_type (glove/word2vec/trainable), --embedding_dim (50/100/200/300)
#   --embedding_trainable
#   --rnn_type (lstm/gru), --hidden_dim, --n_layers, --bidirectional, --dropout
#   --batch_size, --max_seq_length, --lr, --epochs, --val_frequency

EXPERIMENTS = [
    # Experiment 1
    {
        'n_layers': 4
    },
    # Experiment 2
    {
        'dropout': 0.5,
    },
    # Experiment 3
    {
        'dropout': 0,
    },
    # Experiment 4
    {
        'bidirectional': True,
    },
    # Experiment 5
    {
        'embedding_dim': 300
    },
    # Experiment 6
    {
        'embedding_type': 'word2vec'
    }
]

# =============================================================================

def build_command(config: dict, outputs_subdir: str) -> list:
    """Build command line arguments from config dict."""
    cmd = [sys.executable, 'ec_main.py', '--outputs_subdir', outputs_subdir]
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    return cmd

def main():
    print(f"Running {len(EXPERIMENTS)} experiments...")
    print(f"Output directory: ./outputs/{OUTPUTS_SUBDIR}/\n")
    
    for i, config in enumerate(EXPERIMENTS, 1):
        print("=" * 60)
        print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}")
        print(f"Config: {config}")
        print("=" * 60)
        
        cmd = build_command(config, OUTPUTS_SUBDIR)
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nExperiment {i} failed with return code {result.returncode}")
        else:
            print(f"\nExperiment {i} completed successfully")
        
        print("\n")
    
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    
    # Run analyze_golden on the outputs directory
    print("\nGenerating comparison plots and table...")
    subprocess.run([sys.executable, 'analyze_outputs.py', OUTPUTS_SUBDIR])

if __name__ == "__main__":
    main()