import torch
import torch.nn as nn

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