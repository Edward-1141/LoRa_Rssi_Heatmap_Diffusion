import os
import sys

import imageio
import numpy as np
import torch
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Temporary fix to include the app directory
from app.dataset import load_data_and_split
from app.ddpm import DDPM

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        """
        Standard ResNet style convolutional block.
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # Add residual connection, adjusting for channel size if necessary
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414  # Normalization factor
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        """
        Process and downscale the image feature maps.
        """
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        """
        Upsampling using transposed convolution, followed by normalization and activation.
        """
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

    def forward(self, x_c, x_m, skip):
        """
        Parameters:
        - x_c: Context-embedded features
        - x_m: Mask-embedded features
        - skip: Skip connection from the encoder
        """
        # Combine context and mask embeddings
        x = x_c + x_m  # Element-wise addition
        
        # Concatenate with skip connection
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        
        # Apply transposed convolution, normalization, and activation
        x = self.conv(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Generic one-layer Fully Connected (FC) network for embedding.
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=True):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        if is_res and in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.is_res:
            if hasattr(self, 'residual'):
                residual = self.residual(x)
            out += residual
        return out

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)
    
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, condition_dim=3136):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.condition_dim = condition_dim

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(7),
            nn.GELU()
        )

        # Time and Context Embeddings
        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.timeembed3 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(condition_dim, 4 * n_feat)
        self.contextembed2 = EmbedFC(condition_dim, 2 * n_feat)
        self.contextembed3 = EmbedFC(condition_dim, n_feat)

        # Mask Embeddings
        self.maskembed1 = EmbedFC(condition_dim, 4 * n_feat)
        self.maskembed2 = EmbedFC(condition_dim, 2 * n_feat)
        self.maskembed3 = EmbedFC(condition_dim, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 7, 7),
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )

        # **Updated UnetUp Initializations:**
        self.up1 = UnetUp(8 * n_feat, 2 * n_feat)   # 4*n_feat (x_c +x_m) +4*n_feat (skip) =8*n_feat
        self.up2 = UnetUp(4 * n_feat, n_feat)       # 2*n_feat (x_c +x_m) +2*n_feat (skip) =4*n_feat
        self.up3 = UnetUp(2 * n_feat, n_feat)       # n_feat (x_c +x_m) +n_feat (skip) =2*n_feat
        
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, context_mask, t):
        # Initial Convolution
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        # Flatten Context
        batch_size = c.shape[0]
        c = c.view(batch_size, -1)
        context_mask = context_mask.view(batch_size, -1)
        
        # Compute Context Embeddings
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.n_feat, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat, 1, 1)

        # Compute Mask Embeddings
        memb1 = self.maskembed1(context_mask).view(-1, self.n_feat * 4, 1, 1)
        memb2 = self.maskembed2(context_mask).view(-1, self.n_feat * 2, 1, 1)
        memb3 = self.maskembed3(context_mask).view(-1, self.n_feat, 1, 1)

        # Initial Upsampling
        up1 = self.up0(hiddenvec)  # Assume up0 maintains [36,1024,7,7]
        
        # Combine Context and Mask Embeddings for Up1
        up1_c = cemb1 * up1 + temb1  # [36,1024,7,7]
        up1_m = memb1 * up1 + temb1  # [36,1024,7,7]

        up2 = self.up1(up1_c, up1_m, down3)  # Expected to receive [36,2048,7,7]

        # Combine Context and Mask Embeddings for Up2
        up2_c = cemb2 * up2 + temb2  # [36,512,14,14]
        up2_m = memb2 * up2 + temb2  # [36,512,14,14]
        up3 = self.up2(up2_c, up2_m, down2)  # Expected to receive [36,1024,14,14]
        
        # Combine Context and Mask Embeddings for Up3
        up3_c = cemb3 * up3 + temb3  # [36,256,28,28]
        up3_m = memb3 * up3 + temb3  # [36,256,28,28]
        up4 = self.up3(up3_c, up3_m, down1)  # Expected to receive [36,512,28,28]
        out = self.out(up4)  # [36, in_channels, 28,28]
        return out

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
    nn_model = ContextUnet(
        in_channels=1,       # Adjust if different
        n_feat=256,          # Must match the trained model
        condition_dim=3136, # Must match the trained model
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
    for i, data in enumerate(dataloader):

        image, condition, mask = data
        # Move tensors to CPU first to minimize GPU memory usage
        image = image.to('cpu')
        condition = condition.to('cpu')
        mask = mask.to('cpu')
        selected_samples.append((image, condition, mask))
        # Optional: Limit the number of samples to prevent excessive memory usage
        # For example, select only 10 samples
        if i >= 2:
            break

    # Iterate over each selected sample
    for sample_idx, (original_image, condition, original_mask) in enumerate(selected_samples):
        if sample_idx != 2: # only do for the 3rd sample for now
            continue
        # Move tensors to GPU as needed
        original_image = original_image.to(device)
        original_mask = original_mask.to(device)
        original_condition = original_image

        print(f"Sample {sample_idx}: Original Mask Sum: {torch.sum(original_mask)}")
        if torch.sum(original_condition) < 10:
            print(f"Skipping sample {sample_idx} due to low pixel intensity.")
            continue

        # Iterate over each condition size
        all_images = []
        mask = torch.zeros_like(original_condition, dtype=torch.float32)
        for condition_size in conditions_size_list:
            print(f"  Regenerating with condition size = {condition_size} for sample {sample_idx}")
            condition = original_condition.clone().to('cpu') # Copy original condition
            mask = mask.to('cpu') # to cpu for numpy operations

            # # Randomly select nth largest pixels to set to 1
            # condition_flat = condition.flatten()
            # n_largest_indices = np.argpartition(condition_flat, -condition_size)[-condition_size:]
            # mask_flat = mask.flatten()
            # mask_flat[n_largest_indices] = 1
            # mask = mask_flat.reshape(condition.shape)

            # # Randomly select non-zero pixels to set to 1
            # condition_flat = condition.flatten()
            # non_zero_indices = np.where(condition_flat > 0)[0]
            # selected_indices = np.random.choice(non_zero_indices, condition_size, replace=False)
            # mask_flat = mask.flatten()
            # mask_flat[selected_indices] = 1
            # mask = mask_flat.reshape(condition.shape)

            # Randomly select more pixels to set to 1
            condition_flat = condition.flatten()
            mask_flat = mask.flatten()
            existing_indices = np.where(mask_flat > 0)[0]
            non_zero_indices = np.where(condition_flat > 0)[0]
            new_non_zero_indices = np.setdiff1d(non_zero_indices, existing_indices)
            new_selected_indices = np.random.choice(new_non_zero_indices, condition_size - len(existing_indices), replace=False)
            combined_indices = np.concatenate((existing_indices, new_selected_indices))

            mask_flat[combined_indices] = 1
            mask = mask_flat.reshape(condition.shape)

            #print number of 1s in mask
            # print(f"  Number of 1s in mask: {torch.sum(mask)}")

            # Move tensors to GPU as needed
            # mask = torch.tensor(mask, dtype=torch.float32).to(device)
            mask = mask.to(device)
            condition = condition.to(device)

            # Generate new sample
            x_gen, _ = ddpm.sample(
                n_sample=1, 
                condition=condition,  # Use masked condition
                mask=mask,
                device=device, 
                guide_w=guidew
            )

            # Move tensors to CPU and remove batch dimension for visualization
            x_gen_cpu = x_gen.cpu().squeeze(0)
            condition_cpu = condition.cpu().squeeze(0)
            mask_cpu = mask.cpu().squeeze(0)

            # **Append Generated Samples**
            all_images.append(x_gen_cpu)

            # **Append Condition Image and Ground Truth**
            all_images.append(condition_cpu * mask_cpu)  # Masked condition image
            all_images.append(original_image.cpu().squeeze(0))

        images_per_row = 3  # 1 generated sample + condition + ground truth

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
            axs[i, 1].set_title(f"Condition Image (with {conditions_size_list[i]} pixels)")
            axs[i, 2].set_title("Ground Truth Image")

            tmp_gif_path = os.path.join(save_dir, f"temp_frame_{i}.png")
            plt.savefig(tmp_gif_path)
            plt.close()

            gif_images.append(imageio.imread(tmp_gif_path))

        gif_path = f"{save_path[:-4]}.gif"
        imageio.mimsave(gif_path, gif_images)

        plt.savefig(save_path_color)
                                
        print(f"  Saved recall image for sample {sample_idx} with condition size at {save_path_color}")

    print("\nRecall and regeneration completed.")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_ep30000.pth"
    train_dataset, test_dataset = load_data_and_split('heatmap_norm')
    recall_and_regenerate_edward_test(checkpoint_path, test_dataset, device="cuda", save_dir="recall_images_edward_100_700", conditions_size_list=np.arange(10, 100, 20))