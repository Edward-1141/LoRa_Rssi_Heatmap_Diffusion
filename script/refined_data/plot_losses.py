import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def smooth_data(data, window_size=5):
    """
    Apply moving average smoothing to the data
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def remove_outliers(data, threshold=3):
    """
    Remove outliers using z-score method
    """
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > threshold, np.nan, data)

def plot_losses_with_trends(losses, window_size=5):
    """
    Plot training and validation losses with sophisticated smoothing.
    
    Args:
        losses (list): List of dictionaries containing 'train_loss' and 'val_loss'
        window_size (int): Size of the moving average window
    """
    # Set the seaborn style
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))

    # Get the data
    epochs = range(len(losses))
    train_losses = np.array([loss["train_loss"] for loss in losses])
    val_losses = np.array([loss["val_loss"] for loss in losses])

    # Remove outliers
    train_losses_clean = remove_outliers(train_losses)
    val_losses_clean = remove_outliers(val_losses)

    # Plot original data points with reduced opacity
    plt.plot(epochs, train_losses, 'b.', alpha=0.2, label='Training Loss (raw)')
    plt.plot(epochs, val_losses, 'r.', alpha=0.2, label='Validation Loss (raw)')

    # Calculate smoothed trends
    train_smooth = smooth_data(train_losses_clean, window_size)
    val_smooth = smooth_data(val_losses_clean, window_size)
    
    # Adjust epochs for smoothed data
    smooth_epochs = epochs[window_size-1:len(train_smooth)+window_size-1]

    # Plot smoothed trends
    plt.plot(smooth_epochs, train_smooth, 'b-', linewidth=2, label='Training Loss (trend)')
    plt.plot(smooth_epochs, val_smooth, 'r-', linewidth=2, label='Validation Loss (trend)')

    # Set y-axis limits to exclude extreme outliers
    train_95th = np.nanpercentile(train_losses_clean, 95)
    val_95th = np.nanpercentile(val_losses_clean, 95)
    y_max = max(train_95th, val_95th)
    plt.ylim(0, y_max * 1.1)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses with Moving Average Trends")
    plt.legend()

    # Add some styling
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # plot_losses_with_trends(losses)
    pass 