"""Single entrypoint for all operations.

Usage:
    python run.py --train              # Regular training
    python run.py --train --debug      # Debug/overfit-100 sanity test
"""

import argparse
import sys

import hydra
from omegaconf import DictConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Single entrypoint for training operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --train              # Run regular training
    python run.py --train --debug      # Run overfit-100 sanity test
        """,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training (required)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug/overfit mode (use with --train)",
    )

    # Parse known args to allow Hydra overrides to pass through
    args, remaining = parser.parse_known_args()

    if not args.train:
        parser.print_help()
        sys.exit(1)

    # Re-inject remaining args for Hydra
    sys.argv = [sys.argv[0]] + remaining

    return args


# Store parsed args before Hydra takes over
_parsed_args = parse_args()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entrypoint that routes to appropriate training function."""
    if _parsed_args.debug:
        # Force debug mode in config
        cfg.debug.overfit_100 = True
        from landcover.training import debug_train

        debug_train(cfg)
    else:
        from landcover.training import train

        train(cfg)


if __name__ == "__main__":
    main()
