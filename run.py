"""Single entrypoint for all operations.

Usage:
    python run.py --train              # Regular training
    python run.py --train --debug      # Debug/overfit-100 sanity test
"""

import argparse
import sys
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from utils.hydra_config import register_resolvers
from utils.logging import get_logger, setup_run_logging
from utils.run_id import create_run_directory, generate_run_id, get_run_paths


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
        help="Run in debug mode (use with --train)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: auto-generated)",
    )

    # Parse known args to allow Hydra overrides to pass through
    args, remaining = parser.parse_known_args()

    if not args.train:
        parser.print_help()
        sys.exit(1)

    # Re-inject remaining args for Hydra
    sys.argv = [sys.argv[0]] + remaining

    return args


def save_config_and_overrides(
    cfg: DictConfig,
    run_paths: dict[str, Path],
    overrides: list[str],
) -> None:
    """Save resolved config and overrides to run directory.

    Args:
        cfg: Resolved Hydra configuration
        run_paths: Dictionary of run directory paths
        overrides: List of Hydra command-line overrides
    """
    # Save full resolved config
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    run_paths["config_yaml"].write_text(config_yaml)

    # Save overrides
    overrides_text = "\n".join(overrides) if overrides else "# No overrides"
    run_paths["overrides"].write_text(overrides_text)


# Store parsed args before Hydra takes over
_parsed_args = parse_args()

# Generate run_id and create directory structure BEFORE Hydra
_run_id = _parsed_args.run_id or generate_run_id()

# Register custom OmegaConf resolvers
register_resolvers()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entrypoint that routes to appropriate training function."""
    # Get runs_root from config, create run directory
    runs_root = cfg.trainer.get("runs_root", "runs")
    run_dir = create_run_directory(runs_root, _run_id)
    run_paths = get_run_paths(run_dir)

    # Setup logging to run directory
    setup_run_logging(run_dir)
    logger = get_logger(__name__)

    # Print run info
    logger.info("=" * 60)
    logger.info(f"Run ID: {_run_id}")
    logger.info(f"Run Directory: {run_dir}")
    logger.info("=" * 60)

    # Get Hydra overrides
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    overrides = list(hydra_cfg.overrides.task)

    # Save config and overrides
    save_config_and_overrides(cfg, run_paths, overrides)
    logger.info(f"Saved config to {run_paths['config_yaml']}")

    if _parsed_args.debug:
        from landcover.training import debug_train

        debug_train(cfg, _run_id, run_dir)
    else:
        from landcover.training import train

        train(cfg, _run_id, run_dir)


if __name__ == "__main__":
    main()
