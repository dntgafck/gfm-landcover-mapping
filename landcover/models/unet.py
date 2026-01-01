import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard U-Net convolution block: (Conv2d -> Norm -> ReLU) * 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            # Robust group calculation: find largest divisor of out_channels <= groups
            g1 = min(groups, out_channels)
            while out_channels % g1 != 0:
                g1 -= 1
            self.norm1 = nn.GroupNorm(num_groups=g1, num_channels=out_channels)

            self.norm2 = nn.GroupNorm(num_groups=g1, num_channels=out_channels)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class UNetBaseline(nn.Module):
    """
    A simple, strong, thesis-grade U-Net baseline for land-cover semantic segmentation.
    Compatible with torch.onnx.export.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 11,
        base_channels: int = 32,
        num_stages: int = 4,
        norm_type: str = "batch",
        upsample_type: str = "bilinear",
    ):
        super().__init__()
        if num_stages < 2:
            raise ValueError("num_stages must be at least 2")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.upsample_type = upsample_type

        # Encoder path
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        chs = base_channels
        self.encoders.append(ConvBlock(in_channels, chs, norm_type=norm_type))

        for _ in range(num_stages - 1):
            self.downsamples.append(nn.MaxPool2d(kernel_size=2))
            self.encoders.append(ConvBlock(chs, chs * 2, norm_type=norm_type))
            chs *= 2

        # Decoder path
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _ in range(num_stages - 1):
            if upsample_type == "bilinear":
                self.upsamples.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )
            elif upsample_type == "transpose":
                self.upsamples.append(nn.ConvTranspose2d(chs, chs // 2, kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unsupported upsample_type: {upsample_type}")

            # If bilinear, upsample doesn't change channels, so we cat chs + chs//2
            # If transpose, it changes chs to chs//2, so we cat chs//2 + chs//2
            decoder_in_chs = chs + (chs // 2) if upsample_type == "bilinear" else chs
            self.decoders.append(ConvBlock(decoder_in_chs, chs // 2, norm_type=norm_type))
            chs //= 2

        # Final head
        self.head = nn.Conv2d(chs, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation (ONNX friendly)
        torch._assert(x.shape[1] == self.in_channels, f"Expected {self.in_channels} input channels")

        # Encoder
        enc_features = []
        for i in range(self.num_stages):
            x = self.encoders[i](x)
            if i < self.num_stages - 1:
                enc_features.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for i in range(self.num_stages - 1):
            x = self.upsamples[i](x)
            # Pop corresponding feature from encoder for skip connection
            skip = enc_features.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        # Head
        logits = self.head(x)
        return logits
