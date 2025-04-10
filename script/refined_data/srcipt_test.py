import os
import sys

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Temporary fix to include the app directory
from model.dataset import load_data_v2
from model.context_unet import ContextUnetV2
from model.ddpm import DDPM

def recall_and_regenerate_test(
    checkpoint_path, 
    dataset, 
    device="cuda",
    guidew=2.0,
    save_dir="output/refined_data/recall_images",
    heatmap_idx=0,
    conditions_size_list = np.arange(5, 11, 1),
    
):
    torch.cuda.empty_cache()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load the checkpoint on CPU first
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded on CPU.")
    except RuntimeError as e:
        print("Error loading checkpoint:", e)
        return
    
    # Initialize the model
    nn_model = ContextUnetV2(
        in_channels=1,       # Adjust if different
        n_feat=256,          # Must match the trained model
    )

    # Initialize the DDPM
    betas = (1e-4, 0.02)  # Adjust these values if needed
    n_T = 400             # Number of timesteps, adjust as necessary
    ddpm = DDPM(nn_model, betas, n_T, device)
    
    # Load the state dict
    try:
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        print("State dict loaded successfully on CPU.")
    except RuntimeError as e:
        print("Error loading state_dict:", e)
        # Handle nested keys if necessary
        # Example: Strip 'nn_model.' prefix
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('nn_model.'):
                new_key = key[len('nn_model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        ddpm.load_state_dict(new_state_dict, strict=False)
        print("State dict loaded with strict=False on CPU.")

    # Move the model to GPU
    ddpm = ddpm.to(device)
    ddpm.eval()

    # Clear cached GPU memory again
    torch.cuda.empty_cache()

    # Get the ground truth heatmap
    heatmap = dataset[heatmap_idx].unsqueeze(0) # shape: (1, 1, 56, 56)
    
    # Prepare batch processing  
    all_images = []
    mask = torch.zeros_like(heatmap, dtype=torch.float32)
    batch_conditions = []

    # Create batch of conditions
    for condition_size in conditions_size_list:
        print(f"  Preparing condition size = {condition_size} for heatmap {heatmap_idx}")
        condition = heatmap.clone().to('cpu')  # Copy original condition
        mask = mask.to('cpu')  # to cpu for numpy operations

        # Randomly select more pixels to set to 1
        condition_flat = condition.flatten()
        mask_flat = mask.flatten()
        existing_indices = np.where(mask_flat > 0)[0]
        non_zero_indices = np.where(condition_flat > 0)[0]
        new_non_zero_indices = np.setdiff1d(non_zero_indices, existing_indices)
        new_selction_size = min(len(new_non_zero_indices), condition_size - len(existing_indices))
        new_selected_indices = np.random.choice(new_non_zero_indices, new_selction_size, replace=False)
        combined_indices = np.concatenate((existing_indices, new_selected_indices))
        
        mask_flat[combined_indices] = 1
        mask = mask_flat.reshape(condition.shape)
        mask = mask.to(device)
        condition = condition.to(device)
        condition = condition * mask  # Apply mask to condition

        batch_conditions.append((condition, mask))
        
    # Convert lists to tensors
    conditions_batch = torch.stack([item[0] for item in batch_conditions]).view(len(conditions_size_list), 1, 56, 56)
    masks_batch = torch.stack([item[1] for item in batch_conditions]).view(len(conditions_size_list), 1, 56, 56)

    # Generate new samples in batch
    x_gen_batch, _ = ddpm.sample(
        n_sample=len(conditions_size_list),
        condition=conditions_batch,
        mask=masks_batch,
        device=device,
        guide_w=guidew
    )

    # Process and store results
    for i in range(len(conditions_size_list)):
        x_gen_cpu = x_gen_batch[i].cpu()
        condition_cpu = conditions_batch[i].cpu()
        mask_cpu = masks_batch[i].cpu()

        # Append Generated Samples  
        all_images.append(x_gen_cpu)

        # Append Condition Image and Ground Truth
        all_images.append(condition_cpu)  # Masked condition image
        all_images.append(mask_cpu)  # Mask image
        all_images.append(heatmap.cpu().squeeze(0))

    images_per_row = 4  # 1 generated sample + condition + ground truth

    all_images_inverted = [img * -1 + 1 for img in all_images]

    grid = make_grid(all_images_inverted, nrow=images_per_row, padding=2, normalize=False)

    # **Save the grid image**, each row corresponds to a different condition size
    save_path = os.path.join(save_dir, f"heatmap_{heatmap_idx}_eval.png")
    save_image(grid, save_path)
    print(f"  Saved recall image for heatmap {heatmap_idx} at {save_path}")
    
    # Save the grid image with a colormap
    save_path_color = f"{save_path[:-4]}_color.png"
    grid_np = np.array([img.numpy() for img in all_images]).reshape(-1, images_per_row, 56, 56)
    # grid_np = (grid_np - grid_np.min()) / (grid_np.max() - grid_np.min())

    fig, axs = plt.subplots(grid_np.shape[0], grid_np.shape[1], figsize=(10, 20))
    gif_images = []
    for i in range(grid_np.shape[0]):
        gif_fig, gif_ax = plt.subplots(1, grid_np.shape[1], figsize=(6, 2))
        for j in range(grid_np.shape[1]):
            gif_ax[j].imshow(grid_np[i, j] * -1 + 1, cmap='gray')
            gif_ax[j].axis('off')

            grid = grid_np[i, j]
            if j == 3: # Ground Truth Image
                grid[grid <= 0] = np.nan
            axs[i, j].imshow(grid, cmap='viridis')
            axs[i, j].axis('off')
            
            
        tmp_gif_path = os.path.join(save_dir, f"temp_frame_{i}.png")
        plt.savefig(tmp_gif_path)
        plt.close()

        gif_images.append(imageio.v2.imread(tmp_gif_path))

    axs[0, 0].set_title(f"Generated Sample")
    axs[0, 1].set_title(f"Condition")
    axs[0, 2].set_title("Mask Image")
    axs[0, 3].set_title("Ground Truth Image")  
    
    gif_path = f"{save_path[:-4]}.gif"
    imageio.mimsave(gif_path, gif_images)

    plt.savefig(save_path_color)

    # Delete temporary images
    for i in range(grid_np.shape[0]):
        tmp_gif_path = os.path.join(save_dir, f"temp_frame_{i}.png")
        os.remove(tmp_gif_path)
    
    print(f"  Saved recall image for heatmap {heatmap_idx} at {save_path_color}")
    print("\nRecall and regeneration completed.")


if __name__ == "__main__":

    # Load the dataset
    train_dataset, test_dataset = load_data_v2(train_file='data/refined_data/train_heatmap_norm.h5', test_file='data/refined_data/test_heatmap_norm.h5')

    recall_and_regenerate_test(
        checkpoint_path="checkpoints/refined_data/20250403_024824/saved_checkpoint.pth",
        dataset=train_dataset,
        device="cuda:0",
        save_dir="output/refined_data/recall_images",
        guidew=2.0,
        heatmap_idx=7,
        # conditions_size_list=np.arange(5, 11, 1),
        conditions_size_list=np.hstack([np.arange(1, 51, 5), np.arange(51, 401, 50)]).flatten(),
    )