import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_image_similarity_torch(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, List[float]]:
    """
    Args:
        gt: Ground truth tensor (shape: [B, C, H, W] or [C, H, W] or [H, W])
        pred: Predicted tensor (shape: [B, C, H, W] or [C, H, W] or [H, W])
    Returns:
        Dictionary of similarity metrics, where each value is a list of metrics for each image in the batch
    """
    # Ensure both tensors are on the same device
    pred_device = pred.device
    gt = gt.to(pred_device)
    
    gt_4d, pred_4d = _reshape_to_4d(gt), _reshape_to_4d(pred)

    # Handle size mismatch - if gt is a single image and pred is a batch
    if gt_4d.shape[0] == 1 and pred_4d.shape[0] > 1:
        # Repeat the ground truth to match the batch size and give a warning
        print("Warning: gt is a single image and pred is a batch. Repeating gt to match batch size.")
        gt_4d = gt_4d.repeat(pred_4d.shape[0], 1, 1, 1)
    
    batch_size = gt_4d.shape[0]
    
    metrics = {}
    mse_batch = F.mse_loss(gt_4d, pred_4d, reduction='none').mean(dim=[1, 2, 3])
    mae_batch = F.l1_loss(gt_4d, pred_4d, reduction='none').mean(dim=[1, 2, 3])
    metrics['MSE (Lower is better)'] = mse_batch.tolist()
    metrics['MAE (Lower is better)'] = mae_batch.tolist()
    metrics['PSNR (Higher is better)'] = _calculate_psnr_batch(mse_batch).tolist()
    metrics['SSIM (Higher is better)'] = _calculate_ssim_batch(gt_4d, pred_4d)
    
    # Reshape to [B, H*W*C] for easier processing
    flat_gt = gt_4d.view(batch_size, -1)
    flat_pred = pred_4d.view(batch_size, -1)
    metrics['Pearson_r (Higher is better)'] = _calculate_pearson_batch(flat_gt, flat_pred).tolist()
    metrics['Wasserstein (Lower is better)'] = _calculate_wasserstein_batch(flat_gt, flat_pred).tolist()
    metrics['KL_div (Lower is better)'] = _calculate_kl_divergence_batch(flat_gt, flat_pred).tolist()
    
    return metrics

def plot_evaluation_metrics(metrics: Dict[str, List[float]], save_path: str = None, 
                           x_label: str = "Sampled Conditions Size",
                           extend_x_axis = None,
                           show_plot: bool = False) -> None:
    """
    Plot evaluation metrics as line plots.
    
    Args:
        metrics: Dictionary of metrics, where each value is a list of metrics for each image
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        x_label: Label for the x-axis
        extend_x_axis: List of values to use for x-axis labels (if None, use indices)
        show_plot: Whether to display the plot
    """
    # Create figure with subplots
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 3*num_metrics))
    
    # If there's only one metric, make axs a list for consistent indexing
    if num_metrics == 1:
        axs = [axs]
    
    # Create subdirectory for each metric
    save_dir = os.path.dirname(save_path)
    os.makedirs(os.path.join(save_dir, "single_plots"), exist_ok=True)

    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axs[i]
        if extend_x_axis is None:
            x = np.arange(len(values))
        else:
            x = extend_x_axis

        ax.plot(x, values, marker='o', linestyle='-', linewidth=2, markersize=6)

        # Add mean and std annotations
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7)
        ax.text(len(values)-1, mean_val, f'Mean: {mean_val:.4f}', 
                verticalalignment='bottom', horizontalalignment='right')
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}")
        ax.grid(True, linestyle='--', alpha=0.7)


    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def _reshape_to_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 4D: [batch, channel, height, width]"""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor

def _calculate_psnr_batch(mse_batch: torch.Tensor) -> torch.Tensor:
    """Calculate PSNR from MSE values for a batch"""
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse_batch)

def _calculate_ssim_batch(gt: torch.Tensor, pred: torch.Tensor) -> List[float]:
    """Calculate SSIM for a batch of images"""
    batch_size = gt.shape[0]
    ssim_values = []
    
    # Create SSIM module
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(gt.device)
    
    # Calculate SSIM for each image in the batch
    for i in range(batch_size):
        # Extract single image pair
        gt_img = gt[i:i+1]  # Keep batch dimension
        pred_img = pred[i:i+1]
        
        # Calculate SSIM
        ssim_value = ssim_module(pred_img, gt_img).item()
        ssim_values.append(ssim_value)
    
    return ssim_values

def _calculate_pearson_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate Pearson correlation coefficient for a batch"""
    x_centered = x - torch.mean(x, dim=1, keepdim=True)
    y_centered = y - torch.mean(y, dim=1, keepdim=True)
    covariance = torch.mean(x_centered * y_centered, dim=1)
    std_product = torch.std(x, dim=1) * torch.std(y, dim=1)
    return covariance / (std_product + 1e-8)

def _calculate_wasserstein_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate 1D Wasserstein distance between distributions for a batch"""
    # Sort each row independently
    sorted_x = torch.sort(x, dim=1).values
    sorted_y = torch.sort(y, dim=1).values
    return torch.mean(torch.abs(sorted_x - sorted_y), dim=1)

def _calculate_kl_divergence_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between histograms for a batch"""
    # Create histograms for each image in the batch
    batch_size = x.shape[0]
    kl_divs = []
    
    for i in range(batch_size):
        hist_x = torch.histc(x[i], bins=100, min=0, max=1) + 1e-8
        hist_y = torch.histc(y[i], bins=100, min=0, max=1) + 1e-8
        hist_x = hist_x / hist_x.sum()
        hist_y = hist_y / hist_y.sum()
        kl_div = F.kl_div(
            hist_y.log(), 
            hist_x, 
            reduction='batchmean', 
            log_target=False
        )
        kl_divs.append(kl_div)
    
    return torch.tensor(kl_divs, device=x.device)