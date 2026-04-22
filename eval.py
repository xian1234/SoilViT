"""
SoilViT evaluation script.

All hyperparameters are read from a YAML config file (default:
config/defaults.yaml).  Any value can be overridden on the command line:

    python eval.py --config config/defaults.yaml training.save_path=my_ckpt.pth

Usage
-----
    python eval.py                          # use defaults
    python eval.py --config my_config.yaml  # custom config
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from LandVit import GeoDataset, SoilViT


# ---------------------------------------------------------------------------
# Config helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_set(cfg: dict, dotted_key: str, value: str):
    """Set cfg[a][b][c] from dotted key 'a.b.c', auto-casting the value."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    leaf = keys[-1]
    orig = node.get(leaf)
    if orig is None or isinstance(orig, str):
        node[leaf] = value
    elif isinstance(orig, bool):
        node[leaf] = value.lower() in ("1", "true", "yes")
    elif isinstance(orig, int):
        node[leaf] = int(value)
    elif isinstance(orig, float):
        node[leaf] = float(value)
    elif isinstance(orig, list):
        node[leaf] = yaml.safe_load(value)
    else:
        node[leaf] = value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/defaults.yaml",
                   help="Path to YAML config file")
    p.add_argument("overrides", nargs="*",
                   help="key=value pairs to override config, e.g. training.save_path=ckpt.pth")
    return p.parse_args()

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# Loss (mirrors train.py; std_values sourced from config/defaults.yaml)
# ---------------------------------------------------------------------------

class WeightedL1Loss(nn.Module):
    """Inverse-variance weighted L1 loss.  std_values read from config."""

    def __init__(self, std_values: list, device: torch.device):
        super().__init__()
        weights = 1.0 / (torch.tensor(std_values, dtype=torch.float32) + 1e-8)
        weights = weights / weights.sum()
        self.register_buffer("weights", weights.to(device))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(y_pred - y_true)
        return (l1 * self.weights.view(1, -1, 1, 1)).mean()

class Evaluator:
    """
    Evaluator class for model performance assessment and visualization.
    Handles loading model weights, evaluating on a dataset, calculating metrics,
    and generating various plots.
    """
    def __init__(self, model, val_loader, device, model_weights_path=None):
        """
        Initialize the evaluator with model, validation data loader, device, and model weights.

        Args:
            model: The trained model to evaluate.
            val_loader: DataLoader for validation dataset.
            device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
            model_weights_path: Path to the model weights file.
        """
        self.model = model.to(device) # Move model to the specified device
        if model_weights_path and os.path.exists(model_weights_path):
            print(f"Loading model weights from: {model_weights_path}")
            self.model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
        else:
            print(f"Model weights not found at {model_weights_path}. Initializing model with random weights.")
        self.val_loader = val_loader
        self.device = device
        # Define band names for clarity in plots and metrics
        self.bands = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6']
        # Stores the top 5 batches based on R2 and RMSE.
        # Each entry is (metric_value, batch_index, predictions_array, targets_array)
        self.best_batches = {
            'r2': [], # Sorted in descending order of R2
            'rmse': [] # Sorted in ascending order of RMSE
        }
        # A list to store all predictions and targets from the validation set for overall metrics
        self.all_predictions_overall = []
        self.all_targets_overall = []

    def evaluate_all(self, batch_size_for_eval=32):
        """
        Evaluate the model on the entire validation dataset and compute metrics for each band.
        Collects all predictions and targets for overall metrics and identifies best performing batches.

        Args:
            batch_size_for_eval: Number of samples to process in parallel during evaluation.
                                 This is different from the val_loader's batch size.

        Returns:
            tuple:
                metrics: Dictionary containing R2, RMSE, slope, and intercept for each band (overall).
                all_predictions: Concatenated numpy array of all predictions.
                all_targets: Concatenated numpy array of all target values.
        """
        # Create a DataLoader with a potentially larger batch size for efficient evaluation
        # This allows processing more samples at once on the GPU.
        eval_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=batch_size_for_eval,
            shuffle=False,
            num_workers=0, # Use half of CPU cores for data loading
            pin_memory=True # Pin memory to speed up data transfer to GPU
        )

        self.model.eval() # Set model to evaluation mode
        
        # Reset best batches and overall prediction/target lists before a new evaluation
        self.best_batches = {'r2': [], 'rmse': []}
        self.all_predictions_overall = []
        self.all_targets_overall = []

        with torch.no_grad(): # Disable gradient calculation for inference
            for batch_idx, (x, y, lat_lon) in tqdm(enumerate(eval_loader), desc="Evaluating Model"):
                # Move data to the specified device
                x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)

                # Get model predictions
                batch_preds = self.model(x, lat_lon)

                # Move predictions and targets to CPU and convert to NumPy for metric calculation
                batch_preds_np = batch_preds.cpu().numpy()
                batch_targets_np = y.cpu().numpy()

                # Accumulate all predictions and targets for overall metrics
                self.all_predictions_overall.append(batch_preds_np)
                self.all_targets_overall.append(batch_targets_np)

                # Calculate metrics for the current batch and update best batches list
                batch_metrics = self._calculate_batch_metrics(batch_preds_np, batch_targets_np)
                self._update_best_batches(batch_idx, batch_metrics, batch_preds_np, batch_targets_np)

        # Concatenate all collected predictions and targets for overall evaluation
        all_predictions_final = np.concatenate(self.all_predictions_overall, axis=0)
        all_targets_final = np.concatenate(self.all_targets_overall, axis=0)

        # Calculate overall metrics for the entire dataset
        overall_metrics = self._calculate_metrics(all_predictions_final, all_targets_final)

        return overall_metrics, all_predictions_final, all_targets_final

    def _calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics (R2, RMSE, slope, intercept) for each band
        across the entire dataset.

        Args:
            predictions (np.ndarray): Numpy array of model predictions (N, C, H, W).
            targets (np.ndarray): Numpy array of target values (N, C, H, W).

        Returns:
            dict: Dictionary containing metrics for each band.
        """
        metrics = {band: {} for band in self.bands}
        num_bands = predictions.shape[1]

        for band_idx, band in tqdm(enumerate(self.bands), desc="Calculating Overall Metrics for Each Band"):
            # Flatten predictions and target values for the current band
            pred_flat = predictions[:, band_idx, :, :].flatten()
            target_flat = targets[:, band_idx, :, :].flatten()

            # Filter out NaN or infinite values if any (though typically not expected from model output/data)
            valid_indices = np.isfinite(pred_flat) & np.isfinite(target_flat)
            pred_flat = pred_flat[valid_indices]
            target_flat = target_flat[valid_indices]

            if len(pred_flat) == 0:
                print(f"Warning: No valid data points for {band}, skipping metric calculation.")
                metrics[band] = {'r2': np.nan, 'rmse': np.nan, 'slope': np.nan, 'intercept': np.nan}
                continue

            # Calculate R2
            r2 = r2_score(target_flat, pred_flat)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))

            # Calculate linear regression parameters (slope and intercept)
            # Reshape for scikit-learn: (n_samples, 1)
            lr = LinearRegression()
            lr.fit(target_flat.reshape(-1, 1), pred_flat)
            slope = lr.coef_[0]
            intercept = lr.intercept_

            # Store metrics
            metrics[band] = {
                'r2': r2,
                'rmse': rmse,
                'slope': slope,
                'intercept': intercept
            }

        return metrics

    def _calculate_batch_metrics(self, predictions, targets):
        """
        Calculate average R2 and RMSE for a batch across all bands.

        Args:
            predictions (np.ndarray): Numpy array of batch predictions (batch_size, bands, height, width).
            targets (np.ndarray): Numpy array of batch targets (batch_size, bands, height, width).

        Returns:
            dict: Dictionary containing average R2 and RMSE for the batch.
        """
        num_bands = predictions.shape[1]
        r2_list = []
        rmse_list = []

        for band_idx in range(num_bands):
            # Flatten data for the current band across all samples in the batch
            pred_flat = predictions[:, band_idx, :, :].flatten()
            target_flat = targets[:, band_idx, :, :].flatten()

            # Filter out NaN or infinite values
            valid_indices = np.isfinite(pred_flat) & np.isfinite(target_flat)
            pred_flat = pred_flat[valid_indices]
            target_flat = target_flat[valid_indices]

            if len(pred_flat) > 0:
                r2_list.append(r2_score(target_flat, pred_flat))
                rmse_list.append(np.sqrt(mean_squared_error(target_flat, pred_flat)))
            else:
                r2_list.append(np.nan)
                rmse_list.append(np.nan)

        # Return the mean of R2 and RMSE across all bands for the batch
        # Filter out NaNs before calculating mean
        avg_r2 = np.nanmean(r2_list) if r2_list else np.nan
        avg_rmse = np.nanmean(rmse_list) if rmse_list else np.nan

        return {
            'r2': avg_r2,
            'rmse': avg_rmse
        }

    def _update_best_batches(self, batch_idx, batch_metrics, batch_predictions, batch_targets):
        """
        Update the list of best batches based on R2 and RMSE metrics.
        Stores the batch index, metric value, and the actual prediction/target data for visualization.

        Args:
            batch_idx (int): Index of the current batch.
            batch_metrics (dict): Dictionary containing 'r2' and 'rmse' for the current batch.
            batch_predictions (np.ndarray): Predictions for the current batch.
            batch_targets (np.ndarray): Targets for the current batch.
        """
        r2_val = batch_metrics['r2']
        rmse_val = batch_metrics['rmse']

        # Update best R2 batches (higher R2 is better)
        if not math.isnan(r2_val): # Only consider valid R2 values
            if len(self.best_batches['r2']) < 5:
                self.best_batches['r2'].append((r2_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['r2'].sort(key=lambda x: x[0], reverse=True) # Sort descending by R2
            elif r2_val > self.best_batches['r2'][-1][0]: # If current R2 is better than the worst in top 5
                self.best_batches['r2'].pop() # Remove the worst
                self.best_batches['r2'].append((r2_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['r2'].sort(key=lambda x: x[0], reverse=True) # Re-sort

        # Update best RMSE batches (lower RMSE is better)
        if not math.isnan(rmse_val): # Only consider valid RMSE values
            if len(self.best_batches['rmse']) < 5:
                self.best_batches['rmse'].append((rmse_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['rmse'].sort(key=lambda x: x[0]) # Sort ascending by RMSE
            elif rmse_val < self.best_batches['rmse'][-1][0]: # If current RMSE is better than the worst in top 5
                self.best_batches['rmse'].pop() # Remove the worst
                self.best_batches['rmse'].append((rmse_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['rmse'].sort(key=lambda x: x[0]) # Re-sort

    def plot_density_scatter(self, predictions, targets, metrics, output_dir='./results'):
        """
        Plots density scatter plots for each band, showing predictions vs. true values on a log scale.
        Uses kernel density estimation for coloring points, includes y=x line, linear fit line,
        and metric annotations.

        Args:
            predictions (np.ndarray): Overall predictions array.
            targets (np.ndarray): Overall targets array.
            metrics (dict): Dictionary of overall evaluation metrics.
            output_dir (str): Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\nStarting to plot density scatter plots...")

        # Determine total number of elements in a flattened band
        total_elements_per_band = predictions.shape[0] * predictions.shape[2] * predictions.shape[3]

        # Set a maximum number of points for plotting to manage memory and speed.
        max_plot_points = 500_000

        for band_idx, band in tqdm(enumerate(self.bands), desc="Plotting Density Scatter for Each Band"):
            pred_flat = predictions[:, band_idx, :, :].flatten()
            target_flat = targets[:, band_idx, :, :].flatten()

            # Filter out NaN or infinite values before plotting
            valid_indices = np.isfinite(pred_flat) & np.isfinite(target_flat)
            pred_flat = pred_flat[valid_indices]
            target_flat = target_flat[valid_indices]

            if len(pred_flat) == 0:
                print(f"Warning: No valid data points for {band} for scatter plot, skipping.")
                continue

            # --- Sampling for large datasets ---
            if len(pred_flat) > max_plot_points:
                # Generate random indices for sampling without replacement
                indices = np.random.choice(len(pred_flat), max_plot_points, replace=False)
                pred_flat_sampled = pred_flat[indices]
                target_flat_sampled = target_flat[indices]
            else:
                pred_flat_sampled = pred_flat
                target_flat_sampled = target_flat
            # --- End Sampling ---

            plt.figure(figsize=(10, 8))
            
            # Apply log transformation for plotting, handling zeros
            x_log = target_flat_sampled # np.log10(np.maximum(target_flat_sampled, 1e-10)) # Avoid log(0)
            y_log = pred_flat_sampled # np.log10(np.maximum(pred_flat_sampled, 1e-10))
            
            # Calculate 2D density using Gaussian KDE
            xy = np.vstack([x_log, y_log])
            z = stats.gaussian_kde(xy)(xy)
            
            # Sort points by density to ensure high-density points are drawn on top
            idx = z.argsort()
            x_log, y_log, z = x_log[idx], y_log[idx], z[idx]
            
            # Plot density scatter
            sc = plt.scatter(x_log, y_log, c=z, s=5, cmap='viridis', alpha=0.6)
            
            # Add y=x reference line (ideal fit) on log scale
            min_log_val = min(x_log.min(), y_log.min())
            max_log_val = max(x_log.max(), y_log.max())
            plt.plot([min_log_val, max_log_val], [min_log_val, max_log_val], 
                     color='red', linestyle='--', linewidth=1.5, label='y=x (Ideal Fit)')
            
            # Add linear regression fit line on log scale
            lr_log = LinearRegression()
            lr_log.fit(x_log.reshape(-1, 1), y_log)
            slope_log = lr_log.coef_[0]
            intercept_log = lr_log.intercept_
            
            x_fit_log = np.array([min_log_val, max_log_val])
            y_fit_log = slope_log * x_fit_log + intercept_log
            plt.plot(x_fit_log, y_fit_log, color='green', linestyle='-', linewidth=2, 
                     label=f'Log-Linear Fit (log10(Y)={slope_log:.2f}log10(X) + {intercept_log:.2f})')

            # Add color bar for point density
            cbar = plt.colorbar(sc)
            cbar.set_label('Point Density')

            # Get metrics for the current band (these are from original scale data)
            band_metrics = metrics.get(band, {})
            r2 = band_metrics.get('r2', np.nan)
            rmse = band_metrics.get('rmse', np.nan)
            slope = band_metrics.get('slope', np.nan)
            intercept = band_metrics.get('intercept', np.nan)

            # Add metric annotations to the plot (for original scale metrics)
            metric_text = (f'R²: {r2:.4f}\n'
                           f'RMSE: {rmse:.4f}\n'
                           f'Slope (Original): {slope:.4f}\n'
                           f'Intercept (Original): {intercept:.4f}')
            # Position text in the top-left corner
            plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

            plt.xlabel('log10(True Value)')
            plt.ylabel('log10(Predicted Value)')
            plt.title(f'{band} Log-Scale Density Plot: Prediction vs. True Value', fontsize=16)
            plt.legend(loc='lower right') # Adjust legend location to avoid overlapping with text
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout() # Adjust layout to prevent labels from overlapping

            # Save the figure
            file_path = os.path.join(output_dir, f'{band}_log_density_scatter.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {file_path}")

    def visualize_best_batches(self, output_dir='./results/best_batches'):
        """
        Visualize the best batches based on R2 and RMSE metrics.
        Each visualization shows prediction, true value, and relative error for each band.

        Args:
            output_dir (str): Directory to save the visualizations.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\nStarting to visualize best batches...")

        # Process best batches for both R2 and RMSE
        for metric_name in ['r2', 'rmse']:
            metric_dir = os.path.join(output_dir, metric_name.upper())
            os.makedirs(metric_dir, exist_ok=True)
            
            print(f"Visualizing best {metric_name.upper()} batches...")
            # Iterate through the stored best batches
            for i, (metric_value, batch_idx, predictions, targets) in enumerate(self.best_batches[metric_name]):
                # Assume we visualize the first sample from the batch
                # If the batch contains multiple samples, you might want to iterate through them
                # or select a representative one. Here, we take the first.
                sample_idx = 0
                if predictions.shape[0] == 0:
                    print(f"Warning: No valid samples in batch {batch_idx}, skipping visualization.")
                    continue
                
                # Create a 6x3 figure (6 bands, 3 plots per band)
                fig, axes = plt.subplots(6, 3, figsize=(18, 28)) # Adjusted figsize for better readability
                
                for band_idx, band in enumerate(self.bands):
                    # Get prediction, target, and relative error for the current band and sample
                    pred = predictions[sample_idx, band_idx, :, :]
                    target = targets[sample_idx, band_idx, :, :]
                    
                    # Calculate relative error: |pred - target| / (|target| + epsilon) * 100%
                    # Add a small epsilon to target to prevent division by zero
                    rel_error = np.abs(pred - target) / (np.abs(target) + 1e-8) * 100
                    
                    # Clip relative error to a reasonable range for visualization (e.g., 0-100%)
                    # Values above 100% might indicate very small true values or large errors,
                    # but visually clamping helps in understanding typical error distribution.
                    rel_error = np.clip(rel_error, 0, 100) 
                    
                    # Plot Prediction
                    im0 = axes[band_idx, 0].imshow(pred, cmap='viridis', aspect='auto')
                    axes[band_idx, 0].set_title(f'{band} Prediction', fontsize=12)
                    axes[band_idx, 0].axis('off')
                    fig.colorbar(im0, ax=axes[band_idx, 0], fraction=0.046, pad=0.04) # Add color bar

                    # Plot True Value
                    im1 = axes[band_idx, 1].imshow(target, cmap='viridis', aspect='auto')
                    axes[band_idx, 1].set_title(f'{band} True Value', fontsize=12)
                    axes[band_idx, 1].axis('off')
                    fig.colorbar(im1, ax=axes[band_idx, 1], fraction=0.046, pad=0.04) # Add color bar

                    # Plot Relative Error
                    # Use 'Reds' colormap for error, vmin=0, vmax=50 (or 100) for consistent scale
                    im2 = axes[band_idx, 2].imshow(rel_error, cmap='Reds', vmin=0, vmax=50, aspect='auto')
                    axes[band_idx, 2].set_title(f'{band} Relative Error (%)', fontsize=12)
                    axes[band_idx, 2].axis('off')
                    fig.colorbar(im2, ax=axes[band_idx, 2], fraction=0.046, pad=0.04) # Add color bar
                
                # Main title for the entire figure
                fig.suptitle(f'Best Batch {batch_idx} (Sample {sample_idx}) - Based on {metric_name.upper()} ({metric_name.upper()}={metric_value:.4f})', fontsize=18, y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                
                # Save the figure
                file_path = os.path.join(metric_dir, f'best_{metric_name}_{i+1}_batch_{batch_idx}.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {file_path}")

def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        k, v = override.split("=", 1)
        deep_set(cfg, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    model = SoilViT(
        img_size=mcfg["img_size"],
        patch_size=mcfg["patch_size"],
        in_chans=mcfg["in_chans"],
        out_chans=mcfg["out_chans"],
        embed_dim=mcfg["embed_dim"],
        depth=mcfg["depth"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        geo_max_value=mcfg["geo_max_value"],
    )

    model_weights_path = tcfg["save_path"]

    test_dataset = GeoDataset(dcfg["test_json"],
                              output_band_indices=dcfg["output_band_indices"])
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    evaluator = Evaluator(model, test_loader, device, model_weights_path)

    print("\nStarting model evaluation and data collection...")
    overall_metrics, all_predictions, all_targets = evaluator.evaluate_all(batch_size_for_eval=1)

    print("\nOverall Evaluation Metrics:")
    for band, band_metrics in overall_metrics.items():
        print(f"{band}:")
        print(f"  R²: {band_metrics['r2']:.4f}")
        print(f"  RMSE: {band_metrics['rmse']:.4f}")
        print(f"  Slope: {band_metrics['slope']:.4f}")
        print(f"  Intercept: {band_metrics['intercept']:.4f}")

    evaluator.plot_density_scatter(all_predictions, all_targets, overall_metrics)
    evaluator.visualize_best_batches()

    print("\nEvaluation completed. Results saved to 'results' directory.")

if __name__ == "__main__":

    main()
