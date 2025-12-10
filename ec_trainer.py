import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
from utils.utils import get_timestamped_logdir 


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, 
                 criterion: nn.Module, 
                 optimizer: optim.Optimizer,
                 config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Trainer class to handle the training pipeline.
        
        Args:
            model: The neural network model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function (e.g., CrossEntropyLoss).
            optimizer: Optimizer (e.g., Adam).
            config: Dictionary containing all hyperparameters and model configs (for saving).
            device: 'cuda' or 'cpu'.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Create output directory for this run
        self.outputs_dir = get_timestamped_logdir('outputs/experiments')
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Setup Logger
        self.logger = self._setup_logger()
        
        # Metrics storage
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epochs': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')

    def _setup_logger(self):
        """Sets up a logger that writes to both file and console."""
        logger = logging.getLogger('EmotionTrainer')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.outputs_dir, 'training_log.txt'))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def _calculate_accuracy(self, predictions, labels):
        """Helper to calculate accuracy per batch."""
        # Get the index of the max log-probability
        top_p, top_class = predictions.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor)).item()

    def _run_epoch(self, loader, is_train: bool):
        """
        Runs a single epoch for training or validation.
        
        Args:
            loader: The DataLoader to iterate over.
            is_train: Boolean flag (True for training, False for validation).
        
        Returns:
            avg_loss: Average loss for the epoch.
            avg_acc: Average accuracy for the epoch.
            error_analysis: String with error breakdown by class.
        """
        self.model.train(is_train) # Set mode (Train/Eval)
        total_loss = 0
        total_acc = 0
        
        # Error tracking: predicted_counts[i] = how many times class i was predicted wrongly
        #                 missed_counts[i] = how many times class i was the true label but missed
        num_classes = self.config.get('num_classes', 6)
        predicted_counts = [0] * num_classes  # False positives per class
        missed_counts = [0] * num_classes     # False negatives per class
        
        # Enable gradients only if training
        with torch.set_grad_enabled(is_train):
            for batch in loader:
                text, labels, lengths = batch
                text = text.to(self.device)
                labels = labels.to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                
                predictions = self.model(text, lengths)
                loss = self.criterion(predictions, labels)
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_acc += self._calculate_accuracy(predictions, labels)
                
                # --- Track errors ---
                _, pred_classes = predictions.topk(1, dim=1)
                pred_classes = pred_classes.squeeze(1)
                # Find misclassified samples
                wrong_mask = pred_classes != labels
                wrong_preds = pred_classes[wrong_mask]
                wrong_labels = labels[wrong_mask]
                for pred in wrong_preds:
                    predicted_counts[pred.item()] += 1
                for label in wrong_labels:
                    missed_counts[label.item()] += 1
                
        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        
        # Build error analysis string
        total_errors = sum(missed_counts)  # same as sum(predicted_counts)
        mode_str = "Train" if is_train else "Val"
        
        if total_errors > 0:
            pred_pct = [f"{c}({100*c/total_errors:.1f}%)" for c in predicted_counts]
            miss_pct = [f"{c}({100*c/total_errors:.1f}%)" for c in missed_counts]
            error_analysis = (f"   [{mode_str} Errors: {total_errors}] "
                            f"Predicted: {pred_pct} | Missed: {miss_pct}")
        else:
            error_analysis = f"   [{mode_str} Errors: 0] Perfect classification!"
        
        return avg_loss, avg_acc, error_analysis

    def _save_checkpoint(self, path):
        """Saves the model state and configuration."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,  # Critical for reconstructing the model later
            'history': self.history
        }
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def _plot_metrics(self):
        """Generates and saves plots for loss and accuracy."""
        epochs = self.history['epochs']
        
        # Build title from config
        c = self.config
        bi_str = "bi" if c.get('bidirectional', False) else "uni"
        title = (f"{c.get('embedding_type', 'emb')} - {c.get('rnn_type', 'rnn')} - "
                 f"{c.get('batch_size', '?')}B - {c.get('lr', '?')}lr - "
                 f"{c.get('n_layers', '?')}L - {bi_str} - {c.get('dropout', '?')}d")
        
        plt.figure(figsize=(12, 5))
        plt.suptitle(title, fontsize=12, fontweight='bold')
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Acc')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.outputs_dir, 'training_metrics.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Plots saved to {save_path}")

    def fit(self, num_epochs, val_frequency=1):
        """
        Main training loop.
        
        Args:
            num_epochs: Total number of epochs to train.
            val_frequency: Validate every X epochs.
        """
        start_time = time.time()
        
        # Log Run Config
        self.logger.info("Starting Training Run")
        self.logger.info(f"Device: {self.device}")
        self.logger.debug(f"Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Do actual training
            train_loss, train_acc, train_error_analysis = self._run_epoch(self.train_loader, is_train=True)
            
            # Record training metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['epochs'].append(epoch)
            										 
            log_msg = f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            error_log = train_error_analysis
            
            # Validation Loop (every X epochs)
            if epoch % val_frequency == 0:
                val_loss, val_acc, val_error_analysis = self._run_epoch(self.val_loader, is_train=False)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                error_log += "\n" + val_error_analysis
                
                # Track best validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss																					 								  
									   
            else:
                # Fill gaps in history to keep list lengths consistent for plotting
                last_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else 0
                last_val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
                self.history['val_loss'].append(last_val_loss)
                self.history['val_acc'].append(last_val_acc)

            # Update Log
            epoch_time = time.time() - epoch_start
            log_msg += f" | Time: {epoch_time:.1f}s"
            self.logger.info(log_msg)
            self.logger.info(error_log)
            
        # Write Final Results
        total_time = time.time() - start_time
        self.logger.info("=" * 30)
        self.logger.info("Training Finished")
        self.logger.info(f"Total Time: {total_time/60:.2f} minutes")
        self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        # Save final model
        model_path = os.path.join(self.outputs_dir, 'model.pt')
        self._save_checkpoint(model_path)
        
        # Save Plots
        self._plot_metrics()
        
        return self.history