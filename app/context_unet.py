import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

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