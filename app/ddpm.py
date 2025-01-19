import torch
import torch.nn as nn
import numpy as np
import random

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, mask):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(noise, self.nn_model(x_t, c, mask, _ts / self.n_T))


    def sample(self, n_sample, condition, mask, device, guide_w=0.0):

        # Extract size from condition
        size = condition.shape[1:]  # [C, H, W]

        # Initialize x_i with random noise
        # set random seed to random number
        torch.manual_seed(random.randint(0, 100000))
        x_i = torch.randn(n_sample, *size, device=device)  # [B, C, H, W]

        # Ensure condition and mask are on the correct device and dtype
        c_i = condition.to(device=device, dtype=x_i.dtype)  # [B, C, H, W]
        mask = mask.to(device=device, dtype=x_i.dtype)     # [B, C, H, W]

        # Optional: Ensure mask is binary
        mask = (mask > 0.5).float()

        x_i_store = []  # To store intermediate steps for visualization

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i:03}', end='\r')

            # Create timestep tensor
            t_is = torch.full((n_sample, 1, 1, 1), i / self.n_T, device=device, dtype=x_i.dtype)

            # Compute epsilon
            with torch.no_grad():
                # Predict noise residual
                eps = self.nn_model(x_i, c_i, mask, t_is)
                
                # If using classifier-free guidance
                if guide_w > 0.0:
                    # Unconditional prediction: set condition and mask to zeros or a special token
                    # Adjust according to your implementation
                    c_i_uncond = torch.zeros_like(c_i)
                    mask_uncond = torch.zeros_like(mask)
                    eps_uncond = self.nn_model(x_i, c_i_uncond, mask_uncond, t_is)
                    eps = (1 + guide_w) * eps - guide_w * eps_uncond

            # Update x_i using the predicted noise and predefined schedules
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * (torch.randn_like(x_i) if i > 1 else 0)
            )

            # Preserve the non-masked regions of the condition
            # Corrected mask operation: use arithmetic inversion instead of bitwise NOT
            x_i = x_i * (1.0 - mask) + c_i * mask

            # Verify tensor integrity
            if not torch.isfinite(x_i).all():
                raise ValueError(f"Non-finite values detected in x_i at timestep {i}.")

            # Store x_i at specified timesteps
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.cpu().numpy())

            # Optional: Clear CUDA cache if memory usage is a concern
            torch.cuda.empty_cache()

        return x_i, x_i_store