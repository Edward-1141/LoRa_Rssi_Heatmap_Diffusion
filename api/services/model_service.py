import os
import importlib

import torch

from model.ddpm import DDPM
from api.services.config import HEATMAP_MODEL_CONFIG

class ModelService:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_version='v1'):
        """Load a model if not already in memory"""
        if model_version in self.models:
            return self.models[model_version]
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            if model_version not in HEATMAP_MODEL_CONFIG['model_types']:
                raise ValueError(f"Unsupported model version: {model_version}")
            
            model_config = HEATMAP_MODEL_CONFIG['model_types'][model_version]
            checkpoint_path = model_config['checkpoint_path']
            model_class = model_config['model_class']
            model_params = model_config['params']

            # Import the model class
            module_name = model_config['module_name']
            module = importlib.import_module(f"model.{module_name}")
            model_class = getattr(module, model_class)
            model = model_class(**model_params)
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Initialize DDPM
            ddpm = DDPM(
                nn_model=model,
                betas=HEATMAP_MODEL_CONFIG['ddpm_params']['betas'],
                n_T=HEATMAP_MODEL_CONFIG['ddpm_params']['n_T'],
                drop_prob=HEATMAP_MODEL_CONFIG['ddpm_params']['drop_prob'],
                device=self.device
            )
            
            # Load state dict
            ddpm.load_state_dict(checkpoint['model_state_dict'])
            ddpm.eval()
            
            # Cache model
            self.models[model_version] = ddpm
            return ddpm
            
        except Exception as e:
            raise

    def get_available_models(self):
        """Get a list of available trained models"""
        return list(HEATMAP_MODEL_CONFIG['model_types'].keys())
    
    
    def generate_heatmap(
        self, 
        rssi_list,
        model_version='v1',
        guide_weight=2.0
    ):
        """
            Generate a heatmap using the specified model
            Args:
                min_lat (float): Minimum latitude of the area
                rssi_list (list): List of RSSI values in tuples key-value pairs (row, col, rssi (normalized))
                model_version (str): Model version to use
            
            Returns:
                heatmap (np.array): A RSSI heatmap of the area 
        """
        MAX_ROW, MAX_COL = HEATMAP_MODEL_CONFIG['model_types'][model_version]['output_dim']
        
        # Get the model
        ddpm = self.load_model(model_version).to(self.device)

        # Convert RSSI list to tensor based on input arguments
        rssi_condition = torch.zeros(1, 1, 56, 56, dtype=torch.float32) # Store the RSSI values in a tensor
        rssi_mask = torch.zeros(1, 1, 56, 56, dtype=torch.float32) # Store the mask values in a tensor 
        
        for row, col, rssi in rssi_list:
            # Skip if the coordinates are outside the area
            if row < 0 or row >= MAX_ROW or col < 0 or col >= MAX_COL:
                # TODO: Log the number of skipped points for user to adjust the area
                continue

            # Update the tensor values
            rssi_condition[0, 0, row, col] = rssi
            rssi_mask[0, 0, row, col] = 1.0
        
        rssi_condition = rssi_condition.to(self.device)
        rssi_mask = rssi_mask.to(self.device)

        # Generate the heatmap
        heatmap = ddpm.sample(
            n_sample=1,
            condition=rssi_condition,
            mask=rssi_mask,
            device=self.device,
            guide_w=guide_weight
        )

        heatmap = heatmap[0].view(56, 56).detach().cpu().numpy()
        return heatmap

if __name__ == '__main__':
    model_service = ModelService()
    print(model_service.get_available_models())
    heatmap = model_service.generate_heatmap(
        rssi_list=[(10, 10, 0.5), (20, 20, 0.8)],
        model_version='v1'
    )
    print(heatmap.shape)