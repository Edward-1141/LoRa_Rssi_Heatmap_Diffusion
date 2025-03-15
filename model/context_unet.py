import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualConvBlock, UnetDown, UnetUp, EmbedFC, CustomConvEncoderV1

class ContextUnetV1(nn.Module):
    def __init__(self, in_channels, n_feat = 256, condition_dim=3136):
        super(ContextUnetV1, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.condition_dim = condition_dim

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(condition_dim, 2*n_feat)
        self.contextembed2 = EmbedFC(condition_dim, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, context_mask, t):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

class ContextUnetV2(nn.Module):
    def __init__(self, in_channels, n_feat = 256, encoder_type='custom_v1', condition_shape=(56, 56)):
        super(ContextUnetV2, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.drop1 = nn.Dropout2d(0.3)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.drop2 = nn.Dropout2d(0.3)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        if encoder_type == 'custom_v1':
            if condition_shape != (56, 56):
                raise ValueError(f"Condition shape must be (56, 56) for encoder type {encoder_type}")
            self.encoder = CustomConvEncoderV1(in_channels * 2, n_feat // 2)
            self.contextembed1 = EmbedFC(in_channels * 2 * 56 * 56, 2*n_feat)
            self.contextembed2 = EmbedFC(in_channels * 2 * 56 * 56, 1*n_feat)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, context_mask, t):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        # expected shape x: torch.Size([128, 1, 56, 56]), c: torch.Size([128, 1, 56, 56]), context_mask: torch.Size([128, 1, 56, 56]), t: torch.Size([128])

        x = self.init_conv(x)
        down1 = self.down1(x)
        down1 = self.drop1(down1)
        down2 = self.down2(down1)
        down2 = self.drop2(down2)
        hiddenvec = self.to_vec(down2)

        c = torch.cat((c, context_mask), 1)
        c_enc = self.encoder(c)
        
        # embed context, time step
        cemb1 = self.contextembed1(c_enc).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c_enc).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out