SEARCH_CONFIG = {
    "agent_types": {
        "greedy": {
            "name": "Greedy Agent",
            "class": "AgentGreedy",
            "module_name": "agent_greedy",
            "heatmap-model-needed": False,
            "params": {
                "explore": 0.2,
                "decay": 0.2
            }
        },
        "heatmap_greedy": {
            "name": "Heatmap Greedy Agent",
            "class": "AgentHeatmapGreedy",
            "module_name": "agent_heatmap_greedy",
            "heatmap-model-needed": True,
            "params": {
                "explore": 0.2,
                "visited_penalty": 0.5,
                "temperature": 1.0,
                "max_history": 100
            },
        }
    }
}

HEATMAP_MODEL_CONFIG = {
    "model_types": {
        "v1": {
            "name": "LoRa Diffusion V1",
            "checkpoint_path": "v1_checkpoints/checkpoint_ep10.pth",
            "model_class": "ContextUnetV1",
            "module_name": "context_unet",
            "params": {
                "in_channels": 1,
                "n_feat": 256,
                "condition_dim": 3136
            },
            "output_dim": [56, 56] #[rows, cols]
        },
        "v2": {
            "name": "LoRa Diffusion V2",
            "checkpoint_path": "v2_checkpoints/checkpoint_ep10.pth",
            "model_class": "ContextUnetV2",
            "module_name": "context_unet",
            "params": {
                "in_channels": 1,
                "n_feat": 256
            },
            "output_dim": [56, 56] #[rows, cols]
        }
    },
    "ddpm_params": {
        "betas": [1e-4, 0.02],
        "n_T": 400,
        "drop_prob": 0.0
    }
}