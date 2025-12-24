import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')
import os
from LandVit import SoilViT, GeoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import math
import scipy.stats as stats

OUTPUT_DIR = './results'
MAX_POINTS_PER_BAND = 500000
USE_LOG_NORM = True
MIN_COUNT_THRESHOLD = 50
CMAP = 'viridis'

plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': 48,
    'axes.labelsize': 64,
    'axes.titlesize': 72,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 48,
    'figure.titlesize': 80
})

VMIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VMAX = [0.9089999759197235, 0.14403093636035919, 0.5395724141597747, 0.307856582403183, 0.24622973680496216, 0.8320677065849305]

class WeightedL1Loss(nn.Module):
    """
    Custom Weighted L1 Loss module.
    Weights are inversely proportional to the standard deviation of each band.
    """
    def __init__(self, device):
        super(WeightedL1Loss, self).__init__()
        std_values = [0.1768311888563579, 0.025509036759667068,
                      0.11163705662290754, 0.07992872250899273, 0.08078752517773179,
                      0.17430399282526382]
        self.weights = 1.0 / (torch.tensor(std_values) + 1e-8)
        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights.to(device)

    def forward(self, y_pred, y_true):
        """
        Computes the weighted L1 loss.
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.
        Returns:
            torch.Tensor: Weighted L1 loss.
        """
        l1 = torch.abs(y_pred - y_true)
        weighted_l1 = l1 * self.weights.view(1, -1, 1, 1)
        return weighted_l1.mean()

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
        self.model = model.to(device)
        if model_weights_path and os.path.exists(model_weights_path):
            print(f"Loading model weights from: {model_weights_path}")
            self.model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
        else:
            print(f"Model weights not found at {model_weights_path}. Initializing model with random weights.")
        self.val_loader = val_loader
        self.device = device
        self.bands = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6']
        self.best_batches = {
            'r2': [],
            'rmse': []
        }
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
        eval_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=batch_size_for_eval,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.model.eval()
        
        self.best_batches = {'r2': [], 'rmse': []}
        self.all_predictions_overall = []
        self.all_targets_overall = []

        with torch.no_grad():
            for batch_idx, (x, y, lat_lon) in tqdm(enumerate(eval_loader), desc="Evaluating Model"):
                x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)

                batch_preds = self.model(x, lat_lon)

                batch_preds_np = batch_preds.cpu().numpy()
                batch_targets_np = y.cpu().numpy()

                self.all_predictions_overall.append(batch_preds_np)
                self.all_targets_overall.append(batch_targets_np)

                batch_metrics = self._calculate_batch_metrics(batch_preds_np, batch_targets_np)
                self._update_best_batches(batch_idx, batch_metrics, batch_preds_np, batch_targets_np)

        all_predictions_final = np.concatenate(self.all_predictions_overall, axis=0)
        all_targets_final = np.concatenate(self.all_targets_overall, axis=0)

        overall_metrics = self._calculate_metrics(all_predictions_final, all_targets_final)

        return overall_metrics, all_predictions_final, all_targets_final

    def _calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics (R2, RMSE, slope, intercept) for each band
        across the entire dataset.

        Args:
            predictions (np.ndarray): Numpy array of model predictions (N, C, H, W).
            targets (np.ndarray): Numpy array of all target values.
        Returns:
            dict: Metrics for each band.
        """
        metrics = {band: {} for band in self.bands}
        num_bands = predictions.shape[1]
        for band_idx, band in tqdm(enumerate(self.bands), desc="Calculating Overall Metrics for Each Band"):
            pred_flat = predictions[:, band_idx, :, :].flatten()
            target_flat = targets[:, band_idx, :, :].flatten()
            valid_indices = np.isfinite(pred_flat) & np.isfinite(target_flat)
            pred_flat = pred_flat[valid_indices]
            target_flat = target_flat[valid_indices]
            if len(pred_flat) == 0:
                print(f"Warning: No valid data for {band}")
                metrics[band] = {'r2': np.nan, 'rmse': np.nan, 'slope': np.nan, 'intercept': np.nan}
                continue
            r2 = r2_score(target_flat, pred_flat)
            rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
            lr = LinearRegression()
            lr.fit(target_flat.reshape(-1, 1), pred_flat)
            metrics[band] = {'r2': r2, 'rmse': rmse, 'slope': lr.coef_[0], 'intercept': lr.intercept_}
        return metrics

    def _calculate_batch_metrics(self, predictions, targets):
        """
        Calculate mean R2 and RMSE for a single batch across all bands.

        Args:
            predictions (np.ndarray): Batch predictions.
            targets (np.ndarray): Batch targets.

        Returns:
            dict: Mean R2 and RMSE for the batch.
        """
        num_bands = predictions.shape[1]
        r2_list, rmse_list = [], []
        for band_idx in range(num_bands):
            p_f = predictions[:, band_idx, :, :].flatten()
            t_f = targets[:, band_idx, :, :].flatten()
            valid = np.isfinite(p_f) & np.isfinite(t_f)
            p_f, t_f = p_f[valid], t_f[valid]
            if len(p_f) > 0:
                r2_list.append(r2_score(t_f, p_f))
                rmse_list.append(np.sqrt(mean_squared_error(t_f, p_f)))
            else:
                r2_list.append(np.nan)
                rmse_list.append(np.nan)
        return {'r2': np.nanmean(r2_list) if r2_list else np.nan, 'rmse': np.nanmean(rmse_list) if rmse_list else np.nan}

    def _update_best_batches(self, batch_idx, batch_metrics, batch_predictions, batch_targets):
        """
        Update the list of best batches based on batch R2 and RMSE.

        Args:
            batch_idx (int): Current batch index.
            batch_metrics (dict): Metrics for the current batch.
            batch_predictions (np.ndarray): Predictions for the batch.
            batch_targets (np.ndarray): Targets for the batch.
        """
        r2_val, rmse_val = batch_metrics['r2'], batch_metrics['rmse']
        if not math.isnan(r2_val):
            if len(self.best_batches['r2']) < 5:
                self.best_batches['r2'].append((r2_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['r2'].sort(key=lambda x: x[0], reverse=True)
            elif r2_val > self.best_batches['r2'][-1][0]:
                self.best_batches['r2'].pop()
                self.best_batches['r2'].append((r2_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['r2'].sort(key=lambda x: x[0], reverse=True)
        if not math.isnan(rmse_val):
            if len(self.best_batches['rmse']) < 5:
                self.best_batches['rmse'].append((rmse_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['rmse'].sort(key=lambda x: x[0])
            elif rmse_val < self.best_batches['rmse'][-1][0]:
                self.best_batches['rmse'].pop()
                self.best_batches['rmse'].append((rmse_val, batch_idx, batch_predictions, batch_targets))
                self.best_batches['rmse'].sort(key=lambda x: x[0])

    def plot_density_scatter(self, predictions, targets, metrics, output_dir=OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nStarting Hexbin plot generation ...")

        for band_idx, band in tqdm(enumerate(self.bands), desc="Plotting Hexbin Scatter"):
            pred_flat = predictions[:, band_idx, :, :].flatten()
            target_flat = targets[:, band_idx, :, :].flatten()
            valid = np.isfinite(pred_flat) & np.isfinite(target_flat)
            x_data = target_flat[valid]
            y_data = pred_flat[valid]
            if len(x_data) < 100: continue

            if len(x_data) > MAX_POINTS_PER_BAND:
                idx = np.random.choice(len(x_data), MAX_POINTS_PER_BAND, replace=False)
                x_plot, y_plot = x_data[idx], y_data[idx]
            else:
                x_plot, y_plot = x_data, y_data

            plt.figure(figsize=(28, 22))
            data_min, data_max = min(x_plot.min(), y_plot.min()), max(x_plot.max(), y_plot.max())
            margin = (data_max - data_min) * 0.05
            lim_min, lim_max = data_min - margin, data_max + margin

            hb = plt.hexbin(x_plot, y_plot, gridsize=80, bins='log' if USE_LOG_NORM else None, cmap=CMAP, mincnt=1)
            
            counts = hb.get_array()
            mask = counts < MIN_COUNT_THRESHOLD
            counts[mask] = np.nan
            hb.set_array(counts)

            if USE_LOG_NORM:
                valid_max = counts[~np.isnan(counts)].max() if np.any(~np.isnan(counts)) else MIN_COUNT_THRESHOLD
                hb.set_norm(LogNorm(vmin=MIN_COUNT_THRESHOLD, vmax=valid_max))

            plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max)
            cbar = plt.colorbar(hb)
            cbar.set_label('Point Density', fontsize=48)
            cbar.ax.tick_params(labelsize=48)

            plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=5, label='y=x (Ideal Fit)')
            band_m = metrics.get(band, {})
            r2, rmse, slope, intercept = band_m.get('r2', 0), band_m.get('rmse', 0), band_m.get('slope', 1), band_m.get('intercept', 0)
            
            x_fit = np.array([lim_min, lim_max])
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, 'g-', linewidth=4, label=f'Linear Fit (Y={slope:.2f}X + {intercept:.2f})')

            metric_text = (f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nSlope (Full): {slope:.4f}\nIntercept (Full): {intercept:.4f}')
            plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=32, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=1.0', fc='white', alpha=0.9, edgecolor='gray', linewidth=2))

            plt.xlabel('True Value', fontsize=64)
            plt.ylabel('Predicted Value', fontsize=64)
            plt.title(f'{panel_labels[band_idx]} {band} Predicted vs. True Value (Hexbin)', fontsize=72)
            plt.legend(loc='lower right', fontsize=32)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout(pad=5.0)
            plt.savefig(os.path.join(output_dir, f'{band}_hexbin_density.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def visualize_best_batches(self, output_dir=OUTPUT_DIR + '/best_batches'):
        """
        Visualize the best batches based on R2 and RMSE metrics.
        Each visualization shows prediction, true value, and relative error for each band.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\nStarting to visualize best batches...")
        
        for metric_name in ['r2', 'rmse']:
            metric_dir = os.path.join(output_dir, metric_name.upper())
            os.makedirs(metric_dir, exist_ok=True)
            
            print(f"Visualizing best {metric_name.upper()} batches...")
            for i, (metric_value, batch_idx, predictions, targets) in enumerate(self.best_batches[metric_name]):
                sample_idx = 0
                if predictions.shape[0] == 0:
                    print(f"Warning: No valid samples in batch {batch_idx}, skipping visualization.")
                    continue
                
                fig, axes = plt.subplots(6, 3, figsize=(18, 28))
                
                for band_idx, band in enumerate(self.bands):
                    pred = predictions[sample_idx, band_idx, :, :]
                    target = targets[sample_idx, band_idx, :, :]
                    
                    rel_error = np.abs(pred - target) / (np.abs(target) + 1e-8) * 100
                    
                    rel_error = np.clip(rel_error, 0, 100)
                    
                    im0 = axes[band_idx, 0].imshow(pred, cmap='viridis', aspect='auto')
                    axes[band_idx, 0].set_title(f'{band} Prediction', fontsize=12)
                    axes[band_idx, 0].axis('off')
                    fig.colorbar(im0, ax=axes[band_idx, 0], fraction=0.046, pad=0.04)

                    im1 = axes[band_idx, 1].imshow(target, cmap='viridis', aspect='auto')
                    axes[band_idx, 1].set_title(f'{band} True Value', fontsize=12)
                    axes[band_idx, 1].axis('off')
                    fig.colorbar(im1, ax=axes[band_idx, 1], fraction=0.046, pad=0.04)

                    im2 = axes[band_idx, 2].imshow(rel_error, cmap='Reds', vmin=0, vmax=50, aspect='auto')
                    axes[band_idx, 2].set_title(f'{band} Relative Error (%)', fontsize=12)
                    axes[band_idx, 2].axis('off')
                    fig.colorbar(im2, ax=axes[band_idx, 2], fraction=0.046, pad=0.04)
                
                fig.suptitle(f'Best Batch {batch_idx} (Sample {sample_idx}) - Based on {metric_name.upper()} ({metric_name.upper()}={metric_value:.4f})', fontsize=18, y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                file_path = os.path.join(metric_dir, f'best_{metric_name}_{i+1}_batch_{batch_idx}.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {file_path}")

def main():
    """
    Main function to run the model evaluation and visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SoilViT(in_chans=13, out_chans=6, embed_dim=768, depth=12, num_heads=12)
    
    model_weights_path = 'model.pth'

    val_dataset = GeoDataset('eval_patch_data.json')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    evaluator = Evaluator(model, val_loader, device, model_weights_path)
    
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