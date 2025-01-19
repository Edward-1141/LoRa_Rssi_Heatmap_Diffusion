import os
import json

import torch
import torch.nn.functional as F

class CustomDatasetEnhanced(torch.utils.data.Dataset):
    def __init__(self, json_paths, mask_ratio=0.7, shift_up_down=0, shift_left_right=25):
        """
        Initializes the dataset.

        Args:
            json_paths (list): List of paths to JSON files.
            mask_ratio (float): Ratio of pixels to mask out.
            shift_up_down (int): Number of pixels to shift vertically.
            shift_left_right (int): Number of pixels to shift horizontally.
            seed (int, optional): Seed for reproducibility.
        """
        self.json_paths = json_paths
        self.mask_ratio = mask_ratio
        self.shift_up_down = shift_up_down
        self.shift_left_right = shift_left_right

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image, condition, and mask for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (shifted_image, condition, mask)
        """
        json_path = self.json_paths[idx]
        with open(json_path, 'r') as f:
            matrix = json.load(f)
        
        image = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
        
        # Perform shifting and structured masking
        shifted_image, shift_mask = self.shift_tensor_with_mask(
            image, self.shift_up_down, self.shift_left_right
        )
        
        # Initialize mask: 1 for visible regions, 0 for masked regions
        mask = torch.ones_like(shifted_image)
        mask[~shift_mask] = 0.0  # Mask shifted regions

        # Convert mask to float32 for consistency
        mask = mask.float()

        # Calculate the number of additional pixels to mask to reach mask_ratio
        total_elements = mask.numel()
        desired_mask = int(self.mask_ratio * total_elements)
        current_mask = (mask == 0.0).sum().item()
        additional_masks = desired_mask - current_mask

        if additional_masks > 0:
            available = (mask == 1.0).nonzero(as_tuple=False)
            num_available = available.size(0)
            
            if num_available > 0:
                # Limit additional_masks to the number of available pixels to prevent errors
                additional_masks = min(additional_masks, num_available)
                
                # Randomly select indices to mask
                selected_indices = torch.randperm(num_available)[:additional_masks]
                indices = available[selected_indices]
                
                # Set selected mask positions to 0
                mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 0.0

        # Create condition by masking the shifted image
        condition = shifted_image.clone()

        return shifted_image, condition, mask

    def shift_tensor_with_mask(self, tensor, shift_up_down, shift_left_right):
        """
        Shift the tensor and return the shifted tensor and mask.
        Handles both vertical and horizontal shifts with proper clamping.

        Args:
            tensor (torch.Tensor): Input tensor of shape (C, H, W).
            shift_up_down (int): Number of pixels to shift vertically.
            shift_left_right (int): Number of pixels to shift horizontally.

        Returns:
            tuple: (shifted_tensor, mask)
        """
        C, H, W = tensor.shape

        # Clamp shift values to prevent over-shifting
        shift_up_down = max(-H + 1, min(shift_up_down, H - 1))
        shift_left_right = max(-W + 1, min(shift_left_right, W - 1))

        shifted = torch.zeros_like(tensor)
        mask = torch.zeros_like(tensor, dtype=torch.bool)

        # Vertical shift
        if shift_up_down > 0:
            # Shift up
            shifted[:, shift_up_down:, :] = tensor[:, :H - shift_up_down, :]
            mask[:, shift_up_down:, :] = True
        elif shift_up_down < 0:
            # Shift down
            shifted[:, :H + shift_up_down, :] = tensor[:, -shift_up_down:, :]
            mask[:, :H + shift_up_down, :] = True
        else:
            # No vertical shift
            shifted[:, :, :] = tensor[:, :, :]
            mask[:, :, :] = True

        # Horizontal shift
        if shift_left_right > 0:
            # Shift left
            shifted = F.pad(shifted[:, :, shift_left_right:], (0, shift_left_right, 0, 0))
            mask = F.pad(mask[:, :, shift_left_right:], (0, shift_left_right, 0, 0))
        elif shift_left_right < 0:
            # Shift right
            shifted = F.pad(shifted[:, :, :W + shift_left_right], (-shift_left_right, 0, 0, 0))
            mask = F.pad(mask[:, :, :W + shift_left_right], (-shift_left_right, 0, 0, 0))
        # No horizontal shift if shift_left_right == 0

        return shifted, mask
    
def load_data_and_split(directory, test_size=0.2):
    json_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(json_paths)} JSON files in {directory}")
    
    if not json_paths:
        raise ValueError(f"No JSON files found in {directory}")
    
    label_to_paths = {}
    for path in json_paths:
        with open(path, 'r') as f:
            content = json.load(f)
            
            if isinstance(content, list):
                # Assuming the first item in the list contains the label
                first_item = content[0] if content else {}
                if isinstance(first_item, dict):
                    label = first_item.get('label', 0)  # Default label if not found
                else:
                    label = 0  # Default label if the first item is not a dictionary
            else:
                label = 0  # Default label if content is not a list
            
            if label not in label_to_paths:
                label_to_paths[label] = []
            label_to_paths[label].append(path)
    
    for label, paths in label_to_paths.items():
        print(f"Label {label}: {len(paths)} files")
    
    train_paths = []
    test_paths = []
    
    for label, paths in label_to_paths.items():
        split_index = int(len(paths) * (1 - test_size))
        train_paths.extend(paths[:split_index])
        test_paths.extend(paths[split_index:])
    
    print(f"Train set size: {len(train_paths)}")
    print(f"Test set size: {len(test_paths)}")
    
    train_dataset = CustomDatasetEnhanced(train_paths)
    test_dataset = CustomDatasetEnhanced(test_paths)
    
    return train_dataset, test_dataset