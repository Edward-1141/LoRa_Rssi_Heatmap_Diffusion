from datetime import datetime
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Temporary fix to include the app directory
from model.dataset import load_data
from model.context_unet import ContextUnetV2
from model.ddpm import DDPM

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
    if not torch.cuda.is_available():
        # Don't train without GPU, I will want to die instead
        raise ValueError("No GPU available")
    
    # hardcoding these here
    n_epoch = 400
    batch_size = 64
    test_batch_size = 16
    n_T = 400
    device = "cuda:0" 
    n_feat = 256
    lrate = 1e-5
    weight_decay = 5e-5
    save_model = True
    check_dir = 'checkpoints/refined_data/' + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = 'output/refined_data/' + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_interval = 10
    eval_interval = 1
    load_checkpoint_path = None
    num_canvas = 56 # 56x56 grid
    # load_checkpoint_path = 'checkpoints/refined_data/checkpoint_ep1.pth'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    # list of tuples for create random samples map for training 
    # (range_start, range_end) which sample size random from range_start to range_end
    training_samples_list = np.array([ 
        [1, 21],
        [1, 21],
        [21, 101],
        [21, 101],
        [101, 901],
    ])

    test_samples_list = np.array([
        [1, 21],
        [21, 101],
        [101, 901],
    ])

    model = ContextUnetV2(in_channels=1, n_feat=n_feat)
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.85)
    ddpm.to(device)

    if load_checkpoint_path:
        checkpoint = torch.load(load_checkpoint_path)
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_checkpoint_path}")

    train_dataset, test_dataset = load_data(train_file='data/refined_data/train_heatmap_norm.h5', test_file='data/refined_data/test_heatmap_norm.h5')
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
    )
    
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
    )

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate, weight_decay=weight_decay)

    # create the transform for data augmentation
    transform = v2.Compose([
        v2.RandomAffine(degrees=180, translate=(0.2, 0.2), fill=0),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),

    ]).to(device)

    json_path = f'{save_dir}/losses.json'
    with open(json_path, 'w') as f:
        f.write('[\n')
    
    # Initialize loss history lists
    train_loss_history = []
    val_loss_history = []
        
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        # # linear lrate decay
        # optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        losses = {"total": 0, "val": 0}
        bar_length = len(dataloader) * len(training_samples_list)
        pbar = tqdm(total=bar_length, desc="Training Progress")
        for ground_truth in dataloader:
            ground_truth = ground_truth.to(device)
            ground_truth = transform(ground_truth)
            max_num_1s = torch.sum(ground_truth[0][0] > 0).item()

            # get the target points
            target_points = torch.argwhere(ground_truth[0][0] > 0)
            all_indices = np.arange(len(target_points))
            ground_truth = ground_truth.repeat(batch_size, 1, 1, 1)
            for range_start, range_end in training_samples_list:
                # create sample map for training
                sample_map = torch.zeros(batch_size, 1, 56, 56).to(device)
                for i in range(batch_size):
                    num_1s = (np.random.randint(range_start, min(range_end, max_num_1s)))
                    sampled_indices = np.random.choice(all_indices, num_1s, replace=False)
                    indices = target_points[sampled_indices]
                    sample_map[i, 0, indices[:, 0], indices[:, 1]] = 1

                # Get the masked out points by multiplying sample_map with ground_truth
                masked_out_points = sample_map * ground_truth[0][0]
                # train the model
                optim.zero_grad()
                loss = ddpm(
                    x=ground_truth,
                    c=masked_out_points,
                    mask=sample_map
                )
                loss.backward()
                losses["total"] += loss.item()
                optim.step()
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

        pbar.close()
        losses["total"] /= bar_length
        print(f"Loss: {losses['total']:.4f}")
        
        
        # validation
        if ep % eval_interval == 0:
            print("Validating...", end="\r")
            ddpm.eval()
            for ground_truth in test_dataloader:
                ground_truth = ground_truth.to(device)
                ground_truth = transform(ground_truth)
                max_num_1s = torch.sum(ground_truth[0][0] > 0).item()

                # get the target points
                target_points = torch.argwhere(ground_truth[0][0] > 0)
                all_indices = np.arange(len(target_points))
                ground_truth = ground_truth.repeat(test_batch_size, 1, 1, 1)
                for range_start, range_end in test_samples_list:
                    # create sample map for training
                    sample_map = torch.zeros(test_batch_size, 1, 56, 56).to(device)
                    for i in range(test_batch_size):
                        num_1s = (np.random.randint(range_start, min(range_end, max_num_1s)))
                        sampled_indices = np.random.choice(all_indices, num_1s, replace=False)
                        indices = target_points[sampled_indices]
                        sample_map[i, 0, indices[:, 0], indices[:, 1]] = 1

                # Get the masked out points by multiplying sample_map with ground_truth
                masked_out_points = sample_map * ground_truth[0][0]
                # train the model
                optim.zero_grad()
                loss = ddpm(
                    x=ground_truth,
                    c=masked_out_points,
                    mask=sample_map
                )
                loss.backward()
                losses["val"] += loss.item()

            losses["val"] /= len(test_dataloader)
            print(f"Validation Loss: {losses['val']:.4f}")
        
        # Store the losses in history
        train_loss_history.append(losses["total"])
        val_loss_history.append(losses["val"])

        # Save the loss record as json per epoch
        with open(json_path, 'a') as f:
            f.write(f'\t{{"epoch": {ep}, "train_loss": {losses["total"]}, "val_loss": {losses["val"]}}}')
            f.write(',\n' if ep < n_epoch - 1 else '\n]')

        if ep % save_interval == 0 and save_model:
            save_checkpoint(ep, ddpm, optim, losses['total'], check_dir, filename=f'checkpoint_ep{ep}.pth') 

    # Save the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()

if __name__ == "__main__":
    train_custom()

        