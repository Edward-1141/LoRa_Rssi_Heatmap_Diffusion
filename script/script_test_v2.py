import os
import sys

import imageio
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Temporary fix to include the app directory
from model.dataset import load_data_and_split, load_data_v1
from model.context_unet import ContextUnetV2
from model.ddpm import DDPM

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def recall_and_regenerate_edward_test(
    checkpoint_path, 
    dataset, 
    device="cuda",
    guidew=2.0, #, 0.5, 2.0 ,5.0, 20.0], 
    conditions_size_list = np.arange(5, 11, 1), #np.arange(5, 56, 5),
    save_dir="recall_images"
):

    # Set CUDA memory allocation config
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Unregonized config
    
    # Clear cached GPU memory
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

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Select samples from the dataset
    selected_samples = []
    heatmap_idx = 37 * 15 
    num_combination = np.arange(1,2000, 100)

    # selected_idx = [(1, 0), 
    #                 (1, 1), 
    #                 (1, 2), 
    #                 (1, 3), 
    #                 (1, 4), 
    #                 (1, 5)]
    # for gt_idx, mask_idx in selected_idx:
    #     image, condition, mask = dataset.direct_get(gt_idx, mask_idx)
    #     image = image.to('cpu').unsqueeze(0)
    #     condition = condition.to('cpu').unsqueeze(0)
    #     mask = mask.to('cpu').unsqueeze(0)
    #     selected_samples.append((image, condition, mask))
    original_image, original_condition, original_mask = dataset.direct_get(heatmap_idx, 1999)
    original_image = original_image.to('cpu').unsqueeze(0)
    original_condition = original_condition.to('cpu').unsqueeze(0)
    original_mask = original_mask.to('cpu').unsqueeze(0)
    selected_samples.append((original_image, original_condition, original_mask))

    # Iterate over each selected sample
    for sample_idx, (original_image, condition, original_mask) in enumerate(selected_samples):
        original_image = original_image.to(device)
        original_mask = original_mask.to(device)
        original_condition = original_image

        print(f"Sample {sample_idx}: Original Mask Sum: {torch.sum(original_mask)}")
        if torch.sum(original_condition) < 10:
            print(f"Skipping sample {sample_idx} due to low pixel intensity.")
            continue

        # Prepare batch processing
        all_images = []
        mask = torch.zeros_like(original_condition, dtype=torch.float32)
        batch_conditions = []

        # Create batch of conditions
        for idx, combination in enumerate(num_combination):
            print(f"  Preparing batch {idx} for sample {sample_idx}")
            _, condition, mask = dataset.direct_get(heatmap_idx, combination)
            condition = condition.to(device).unsqueeze(0)
            mask = mask.to(device).unsqueeze(0)
            print(f"  condtion shape: {condition.shape}, mask shape: {mask.shape}")
            batch_conditions.append((condition, mask))

        # Convert lists to tensors
        conditions_batch = torch.stack([item[0] for item in batch_conditions]).view(len(num_combination), 1, 56, 56)
        masks_batch = torch.stack([item[1] for item in batch_conditions]).view(len(num_combination), 1, 56, 56)

        # Generate new samples in batch
        x_gen_batch, _ = ddpm.sample(
            n_sample=len(num_combination),
            condition=conditions_batch,
            mask=masks_batch,
            device=device,
            guide_w=guidew
        )

        # Process and store results
        for i in range(len(num_combination)):
            x_gen_cpu = x_gen_batch[i].cpu()
            condition_cpu = conditions_batch[i].cpu()
            mask_cpu = masks_batch[i].cpu()

            # Append Generated Samples
            all_images.append(x_gen_cpu)

            # Append Condition Image and Ground Truth
            all_images.append(condition_cpu)  # Masked condition image
            all_images.append(mask_cpu)  # Mask image
            all_images.append(original_image.cpu().squeeze(0))

        images_per_row = 4  # 1 generated sample + condition + ground truth

        all_images_inverted = [img * -1 + 1 for img in all_images]
        grid = make_grid(all_images_inverted, nrow=images_per_row, padding=2, normalize=False)
        
       # **Save the grid image**, each row corresponds to a different condition size
        save_path = os.path.join(save_dir, f"sample_{sample_idx}_condition_size.png")
        save_image(grid, save_path)
        print(f"  Saved recall image for sample {sample_idx} with condition size at {save_path}")

        # Save the grid image with a colormap
        save_path_color = f"{save_path[:-4]}_color.png"
        # grid_np = np.array(all_images).reshape(-1, images_per_row, 56, 56)
        grid_np = np.array([img.numpy() for img in all_images]).reshape(-1, images_per_row, 56, 56)
        grid_np = (grid_np - grid_np.min()) / (grid_np.max() - grid_np.min())

        fig, axs = plt.subplots(grid_np.shape[0], grid_np.shape[1], figsize=(20, 20))
        gif_images = []
        for i in range(grid_np.shape[0]):
            gif_fig, gif_ax = plt.subplots(1, grid_np.shape[1], figsize=(20, 20))
            for j in range(grid_np.shape[1]):
                axs[i, j].imshow(grid_np[i, j], cmap='viridis')
                axs[i, j].axis('off')
                gif_ax[j].imshow(grid_np[i, j] * -1 + 1, cmap='gray')
                gif_ax[j].axis('off')
            
            axs[i, 0].set_title(f"Generated Sample {i+1}")
            axs[i, 1].set_title(f"Condition Image (with {np.sum(masks_batch[i].cpu().numpy())} pixels)")
            axs[i, 2].set_title("Mask Image")
            axs[i, 3].set_title("Ground Truth Image")

            tmp_gif_path = os.path.join(save_dir, f"temp_frame_{i}.png") # save to cache instead
            plt.savefig(tmp_gif_path)
            plt.close()

            gif_images.append(imageio.v2.imread(tmp_gif_path))

        gif_path = f"{save_path[:-4]}.gif"
        imageio.mimsave(gif_path, gif_images)

        plt.savefig(save_path_color)

        # Delete temporary images
        for i in range(grid_np.shape[0]):
            tmp_gif_path = os.path.join(save_dir, f"temp_frame_{i}.png")
            os.remove(tmp_gif_path)
                                
        print(f"  Saved recall image for sample {sample_idx} with condition size at {save_path_color}")

    print("\nRecall and regeneration completed.")

if __name__ == "__main__":
    # checkpoint_path = "checkpoints/checkpoint_ep6000.pth"
    checkpoint_path = "v2_checkpoints/checkpoint_ep14.pth"
    # train_dataset, test_dataset = load_data_and_split('heatmap_norm')
    train_dataset, test_dataset = load_data_v1(train_file='data/train_heatmap_norm_small.h5', test_file='data/test_heatmap_norm_small.h5')
    recall_and_regenerate_edward_test(checkpoint_path, train_dataset, device="cuda", save_dir="output/recall_images")