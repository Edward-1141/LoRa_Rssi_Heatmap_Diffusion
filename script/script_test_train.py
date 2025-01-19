import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Temporary fix to include the app directory
from app.dataset import load_data_and_split
from app.context_unet import ContextUnet
from app.ddpm import DDPM

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_checkpoint(epoch, model, optimizer, loss, save_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

def train_custom():
    # hardcoding these here
    n_epoch = 30001
    batch_size = 256
    n_T = 400
    condition_dim = 3136
    device = "cuda:0"
    n_feat = 256
    lrate = 1e-4
    save_model = True
    save_dir = 'output/diffusion_outputs/'
    check_dir = 'checkpoints/'
    ws_test = [0.0, 2.0, 5.0, 7.5, 10.0, 20.0]  
    val_ep = 6000
    num_train_samples = 2 # Number of samples to use from training set for evaluation
    
    model = ContextUnet(in_channels=1, n_feat=n_feat, condition_dim=condition_dim) 
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.85)
    ddpm.to(device)

    train_dataset, test_dataset = load_data_and_split('heatmap_norm')
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Randomly select samples for evaluation
    train_eval_indices = random.sample(range(len(train_dataset)), num_train_samples)
    test_eval_indices = random.sample(range(len(test_dataset)), len(test_dataset))

    train_eval_samples = [train_dataset[i][0].unsqueeze(0).to(device) for i in train_eval_indices]
    test_eval_samples = [test_dataset[i][0].unsqueeze(0).to(device) for i in test_eval_indices]

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    loss_values = []
    train_eval_loss_values = []
    test_eval_loss_values = []

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c, mask in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            mask = mask.to(device)
            loss = ddpm(x, c, mask)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        loss_values.append(loss_ema)

        if ep % 1000 == 0 and save_model:
            save_checkpoint(ep, ddpm, optim, loss_ema, check_dir, filename=f'checkpoint_ep{ep}.pth')
        
        # Evaluation code
        if ep % val_ep == 0 and ep > 0:
            ddpm.eval()  # Set the model to evaluation mode
            train_eval_loss_temp = []
            test_eval_loss_temp = []
            
            # Pre-generate masks and masked samples for all evaluation samples
            masks = []
            masked_samples = []
            for sample in train_eval_samples + test_eval_samples:
                # Move sample to the correct device and ensure proper dtype
                sample = sample.to(device=device, dtype=torch.float32)
                
                # Create a binary mask with 10% masked (value=1.0) and 90% unmasked (value=0.0)
                mask = (torch.rand(sample.shape, device=device) > 0.9).float()
                
                masks.append(mask)
                # Define masked_sample: preserve unmasked regions, zero out masked regions
                masked_sample = sample * mask
                masked_samples.append(masked_sample)
            
            # Iterate over each guidance weight
            for w_i, w in enumerate(ws_test):
                all_samples = []
                
                for i, (original_sample, mask, masked_sample) in enumerate(zip(train_eval_samples + test_eval_samples, masks, masked_samples)):
                    # Ensure masked_sample is on the correct device and dtype
                    masked_sample = masked_sample.to(device=device, dtype=torch.float32)
                    mask = mask.to(device=device, dtype=torch.float32)
                    
                    # Generate new sample based on the masked_sample and current guidance weight
                    # Note: Assuming `ddpm.sample` is defined as:
                    # def sample(self, n_sample, condition, mask, device, guide_w=0.0):
                    # where `condition` is the masked input, and `mask` indicates regions to generate
                    x_gen, _ = ddpm.sample(n_sample=1, condition=original_sample, mask=mask, device=device, guide_w=w)
                    
                    # Move tensors to CPU for loss computation and visualization
                    x_gen = x_gen.cpu()
                    original_sample_cpu = original_sample.cpu()
                    mask_cpu = mask.cpu()
                    masked_sample_cpu = masked_sample.cpu()
                    
                    # Convert mask to boolean for indexing
                    mask_bool = mask_cpu.bool()
                    
                    # Calculate loss only for masked regions (where mask=1.0)
                    if i < num_train_samples:
                        # Training samples
                        train_loss = F.mse_loss(x_gen[mask_bool], original_sample_cpu[mask_bool])
                        train_eval_loss_temp.append(train_loss.item())
                    else:
                        # Testing samples
                        test_loss = F.mse_loss(x_gen[mask_bool], original_sample_cpu[mask_bool])
                        test_eval_loss_temp.append(test_loss.item())
                    
                    # Prepare samples for visualization
                    # Arrange as [Generated Image, Real * (1 - Mask), Real Image]
                    all_samples.extend([x_gen, masked_sample_cpu, original_sample_cpu])
                
                # Combine all samples into a single tensor for visualization
                # Shape after concat: [3 * num_samples, C, H, W]
                x_all = torch.cat(all_samples)
                
                # Create a grid of images
                # Adjust normalization as needed
                grid = make_grid(x_all * -1 + 1, nrow=3)  # 3 images per row: gen, real*(1-mask), real
                
                # Enlarge the grid image for better visibility
                grid = F.interpolate(grid.unsqueeze(0), scale_factor=5, mode='nearest').squeeze(0)
                
                # Define the save path with epoch and guidance weight information
                save_path = f"{save_dir}/image_ep{ep}_w{w}.png"
                
                # Save the grid image
                save_image(grid, save_path)
                print(f"Saved image at {save_path}")
            
            # Compute average training and testing evaluation losses
            if train_eval_loss_temp:
                avg_train_loss = sum(train_eval_loss_temp) / len(train_eval_loss_temp)
            else:
                avg_train_loss = 0.0
            
            if test_eval_loss_temp:
                avg_test_loss = sum(test_eval_loss_temp) / len(test_eval_loss_temp)
            else:
                avg_test_loss = 0.0
            
            train_eval_loss_values.append(avg_train_loss)
            test_eval_loss_values.append(avg_test_loss)


    # Create an array of epoch numbers where validation occurred
    val_epochs = range(0, n_epoch, val_ep)[:num_val_points]
    # Handle zero or negative loss values by adding a small epsilon
    epsilon = 1e-8
    test_eval_loss_values = [loss + epsilon for loss in test_eval_loss_values]
    train_eval_loss_values = [loss + epsilon for loss in train_eval_loss_values]
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_epoch), loss_values, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()

    num_val_points = len(test_eval_loss_values)

    # Create an array of epoch numbers where validation occurred
    val_epochs = list(range(0, n_epoch, val_ep))[:num_val_points]

    # Validation (Test) Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(val_epochs, test_eval_loss_values, label='Test Sample Loss')
    plt.title('Test Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    def format_func(value, tick_number):
        if value == 0:
            return '0'
        return f'10^{int(np.log10(value))}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    plt.savefig(os.path.join(save_dir, 'test_loss_curve.png'))
    plt.close()

    # Training Evaluation Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(val_epochs, train_eval_loss_values, label='Train Evaluation Loss')
    plt.title('Train Evaluation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    plt.savefig(os.path.join(save_dir, 'train_eval_loss_curve.png'))
    plt.close()

if __name__ == "__main__":
    train_custom()