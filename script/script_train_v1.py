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
from model.dataset import load_data_v1
from model.context_unet import ContextUnet, ContextUnetV1
from model.ddpm import DDPM

# # Set random seed
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
    n_epoch = 50
    batch_size = 128
    n_T = 400
    condition_dim = 3136
    device = "cuda:0"
    n_feat = 256
    lrate = 5e-5
    weight_decay = 1e-5
    save_model = True
    check_dir = 'v1_checkpoints/'
    save_dir = 'output/diffusion_v1/'
    save_interval = 5
    load_checkpoint_path = None
    # load_checkpoint_path = 'v1_checkpoints/checkpoint_ep0.pth'

    model = ContextUnetV1(in_channels=1, n_feat=n_feat, condition_dim=condition_dim) 
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.85)
    ddpm.to(device)

    if load_checkpoint_path:
        checkpoint = torch.load(load_checkpoint_path)
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_checkpoint_path}")

    train_dataset, test_dataset = load_data_v1(train_file='data/train_heatmap_norm.h5', test_file='data/test_heatmap_norm.h5')
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
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate, weight_decay=weight_decay)


    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        losses = {"total": 0, "val": 0}
        pbar = tqdm(dataloader)
        for x, c, mask in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            mask = mask.to(device)
            # # Save x, c, mask as image for debugging
            # fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            # ax[0].imshow(x[0, 0].cpu().numpy(), cmap='viridis')
            # ax[0].set_title('x')
            # ax[1].imshow(c[0, 0].cpu().numpy(), cmap='viridis')
            # ax[1].set_title('c')
            # ax[2].imshow(mask[0, 0].cpu().numpy(), cmap='viridis')
            # ax[2].set_title('mask')
            # plt.savefig(f'{save_dir}/input.png')
            # exit()
            loss = ddpm(x, c, mask)
            loss.backward()
            
            losses["total"] += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            optim.step()
        losses["total"] /= len(dataloader)
        pbar.close()
        
        # validation
        ddpm.eval()
        val_pbar = tqdm(test_dataloader)
        for x, c, mask in val_pbar:
            x = x.to(device)
            c = c.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                loss = ddpm(x, c, mask)
                losses["val"] += loss.item()
        losses["val"] /= len(test_dataloader)
        string = "".join([f"{key} loss: {value:.3f}\t" for key, value in losses.items()])
        print(f"Epoch {ep+1}/{n_epoch}\t" + string)

        if ep % save_interval == 0 and save_model:
            save_checkpoint(ep, ddpm, optim, losses['total'], check_dir, filename=f'checkpoint_ep{ep}.pth')
    
    # Save the loss plot
    plt.plot(losses['total'], label='train')
    plt.plot(losses['val'], label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()

if __name__ == "__main__":
    train_custom()