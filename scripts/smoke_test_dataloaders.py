import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure root (current dir) is in path so we can import landcover
sys.path.append(os.getcwd())

from landcover.datasets.datamodule import LandCoverDataModule
from utils.logging import setup_logging


def denormalize(img_tensor, mean, std):
    """simple denorm for viz"""
    # img_tensor: (C, H, W)
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img_tensor * std + mean


def main():
    setup_logging()
    print("Initializing Smoke Test...")

    # Config
    index_path = "data/index/dataset_index_with_split.csv"
    norm_stats_path = "data/stats/norm_stats.json"

    dm = LandCoverDataModule(
        index_path=index_path,
        norm_stats_path=norm_stats_path,
        batch_size=4,
        num_workers=0,  # Debug on main thread
        test_apply_cloud_filter=True,
        augment=True,
    )

    print("Calling dm.setup()...")
    dm.setup()

    splits = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test (list)": dm.test_dataloader(),  # this returns a list
    }

    # Handle test list
    test_loaders = splits["test (list)"]
    splits["test_iid"] = test_loaders[0]
    if len(test_loaders) > 1:
        splits["test_ood"] = test_loaders[1]
    del splits["test (list)"]

    output_dir = "outputs/smoke_test"
    os.makedirs(output_dir, exist_ok=True)

    for name, loader in splits.items():
        print(f"\n--- Testing Split: {name} ---")
        if len(loader) == 0:
            print("EMPTY LOADER")
            continue

        try:
            batch = next(iter(loader))
            images = batch["image"]
            masks = batch["mask"]
            ids = batch["patch_id"]
            country = batch["country"]

            print(f"Batch Shape: X={images.shape}, Y={masks.shape}")
            print(f"Dtypes: X={images.dtype}, Y={masks.dtype}")
            print(f"Range X: {images.min():.3f} to {images.max():.3f}")
            print(f"Range Y: {masks.min()} to {masks.max()} (Classes: {torch.unique(masks)})")
            print(f"Sample IDs: {ids}")
            print(f"Countries: {country}")

            # Sanity checks
            assert images.ndim == 4
            assert masks.ndim == 3
            assert not torch.isnan(images).any()

            # Visualize first item
            img = images[0]  # C,H,W
            msk = masks[0]  # H,W

            # Get mean/std from dataset
            ds = loader.dataset
            mean = ds.mean
            std = ds.std

            # Denorm
            img_raw = denormalize(img, mean, std)
            # RGB = B04, B03, B02 -> indices 2, 1, 0
            # Assuming bands order B02, B03, B04, B08 from stats file
            # B02=0, B03=1, B04=2, B08=3
            rgb = img_raw[[2, 1, 0], :, :]
            rgb = rgb.permute(1, 2, 0).numpy()

            # Clip for display
            rgb = np.clip(rgb, 0, 0.3) / 0.3  # Simple brightness scaling

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(rgb)
            plt.title(f"{name} Image (RGB)")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(msk.numpy(), cmap="tab20")
            plt.title(f"{name} Label")
            plt.axis("off")

            save_path = os.path.join(output_dir, f"viz_{name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to {save_path}")

        except Exception as e:
            print(f"FAILED to iterate {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\nSmoke Test Complete.")


if __name__ == "__main__":
    main()
