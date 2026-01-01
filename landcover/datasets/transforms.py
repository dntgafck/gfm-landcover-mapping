import random
from typing import Any

import torch


class LandCoverAugmentations:
    """
    Augmentation pipeline for Land Cover patches.
    Applies consistent transformations to both image (x) and mask (y).
    """

    def __init__(self, key_image: str = "image", key_mask: str = "mask"):
        self.key_image = key_image
        self.key_mask = key_mask

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Apply random keys transformations.
        Args:
            sample: Dict containing at least 'image' and 'mask' tensors.
                   image: (C, H, W)
                   mask: (H, W)
        """
        image = sample[self.key_image]
        mask = sample[self.key_mask]

        # Random Horizontal Flip
        if random.random() > 0.5:
            image = torch.flip(image, [-1])
            mask = torch.flip(mask, [-1])

        # Random Vertical Flip
        if random.random() > 0.5:
            image = torch.flip(image, [-2])
            mask = torch.flip(mask, [-2])

        # Random 90-degree Rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        sample[self.key_image] = image
        sample[self.key_mask] = mask
        return sample


def normalize_image(image: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """
    Normalize image tensor channel-wise: (x - mean) / std.
    Args:
        image: (C, H, W) float tensor
        mean: List of means per channel
        std: List of stds per channel
    Returns:
        Normalized tensor of same shape
    """
    if image.shape[0] != len(mean) or image.shape[0] != len(std):
        raise ValueError(
            f"Image channels ({image.shape[0]}) do not match mean/std length ({len(mean)})."
        )

    # Convert lists to tensors for broadcasting [C, 1, 1]
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)

    return (image - mean_tensor) / std_tensor
