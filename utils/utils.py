import os
import socket
import shutil
import matplotlib.pyplot as plt
import torch
import yaml
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import re

def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

def get_project_root():
    """
    Returns the absolute path to the folder where this function is defined.
    This should be placed in a script located at the project root.
    """
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_timestamp_isr():
    return datetime.now(timezone(timedelta(hours=3))).strftime("%b%d_%H-%M-%S")

def get_timestamped_logdir(subdir_name="runs"):
    """Generate a full log_dir path in main script's directory with Israel timezone timestamp."""
    hostname = socket.gethostname()
    # Get timestamp in Israel timezone
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%b%d_%H-%M-%S")
    # Path to the main script (not where it's called from)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Combine path
    log_dir = os.path.join(base_path, subdir_name, f"{timestamp}_{hostname}")
    return log_dir


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def is_main_process(ddp_enabled):
    # Check if the current process is the main process (in case of DDP)
    return not ddp_enabled  or torch.distributed.get_rank() == 0

def save_metric_old(metric_per_epoch, metric_name, save_path, apply_log=False):
    """
    Saves a graph of a given metric (e.g., loss or accuracy) over training epochs.
    Optionally applies natural logarithm to the metric before plotting.

    Args:
        metric_per_epoch (list or array-like): Values of the metric for each epoch.
        metric_name (str): Name of the metric (used in title and filename).
        save_path (str): Directory path to save the plot. File will be named as '<metric_name>.png'.
        apply_log (bool): Whether to apply natural log to the metric values before plotting.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Apply log if needed
    if apply_log:
        metric_vals = np.log(np.array(metric_per_epoch))
        name_suffix = f"log_{metric_name}"
        title = f'Log({metric_name}) per Epoch'
    else:
        metric_vals = metric_per_epoch
        name_suffix = metric_name
        title = f'{metric_name.capitalize()} per Epoch'

    # Create and save plot
    plot_filename = os.path.join(save_path, f"{name_suffix}.png")
    plt.figure()
    plt.plot(range(1, len(metric_vals) + 1), metric_vals, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(name_suffix)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def save_metric(metric_per_epoch, metric_name, save_path, apply_log=False, ditch_first_10=False):
    """
    Saves a graph of training metrics over epochs.
    If input is a list, plots a single metric.
    If input is a dictionary, plots multiple metrics on the same graph.

    Args:
        metric_per_epoch (list or dict): List of metric values or a dict of lists keyed by metric names.
        metric_name (str): Base name for the plot and saved file.
        save_path (str): Directory to save the plot.
        apply_log (bool): Whether to apply natural log to the metric values before plotting.
        ditch_first_10 (bool): If True, skip the first 10 epochs in the plot for better visualization.
    """
    os.makedirs(save_path, exist_ok=True)
    plt.figure()

    if isinstance(metric_per_epoch, dict):
        for key, values in metric_per_epoch.items():
            # Skip first 10 epochs if requested
            if ditch_first_10 and len(values) > 10:
                values = values[10:]
                epoch_start = 11  # Start from epoch 11
            else:
                epoch_start = 1
            
            if apply_log:
                values = np.log(np.array(values))
            
            plt.plot(range(epoch_start, epoch_start + len(values)), values, marker='o', label=key)
        plt.legend()
    else:
        # Skip first 10 epochs if requested
        if ditch_first_10 and len(metric_per_epoch) > 10:
            values = metric_per_epoch[10:]
            epoch_start = 11  # Start from epoch 11
        else:
            values = metric_per_epoch
            epoch_start = 1
            
        values = np.log(np.array(values)) if apply_log else values
        plt.plot(range(epoch_start, epoch_start + len(values)), values, marker='o')

    # Update title and filename to reflect if first epochs were skipped
    skip_text = " (First 10 Epochs Skipped)" if ditch_first_10 else ""
    title = f"{'Log ' if apply_log else ''}{metric_name.capitalize()} per Epoch{skip_text}"
    ylabel = "Log Loss" if apply_log else "Loss"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    # Add suffix to filename if first epochs were skipped
    filename_suffix = "_skip10" if ditch_first_10 else ""
    full_path = os.path.join(save_path, f"{metric_name}{filename_suffix}.png")
    plt.savefig(full_path)
    plt.close()

def extract_losses_from_log(log_path):
    """
    Extracts loss values from a log file with lines containing 'Loss: <value>'.
    """
    with open(log_path, "r") as f:
        content = f.read()
    losses = [float(match) for match in re.findall(r"Loss:\s+([0-9.]+)", content)]
    return losses

def get_israel_time():
    tz = pytz.timezone("Asia/Jerusalem")
    return datetime.now(tz)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
