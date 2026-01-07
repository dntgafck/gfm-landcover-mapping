"""Single entrypoint for all operations.

Usage:
    python run.py train                                    # Regular training
    python run.py train run_id=my-custom-id                # Training with custom run ID
    python run.py train trainer.max_epochs=100             # Training with Hydra overrides
    python run.py train run_id=my-id trainer.max_epochs=100  # Multiple overrides
    python run.py debug                                    # Debug/overfit-100 sanity test
    python run.py export <run_id>                          # Export model from existing run
    python run.py export <run_id> --checkpoint=last        # Export using last checkpoint
"""

import sys
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from utils.hydra_config import register_resolvers
from utils.logging import get_logger, setup_run_logging
from utils.run_id import create_run_directory, generate_run_id, get_run_paths

# Register custom OmegaConf resolvers
register_resolvers()

# Absolute path to configs directory
CONFIG_DIR = str(Path(__file__).parent / "configs")


def _load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra config using Compose API.

    Args:
        config_name: Name of the config file (without .yaml)
        overrides: List of Hydra-style overrides (e.g., ["trainer.max_epochs=100"])

    Returns:
        Loaded and resolved configuration
    """
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    # Initialize and compose
    initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3")
    cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def _save_config_and_overrides(
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


class CLI:
    """Command-line interface for training and export operations.

    Available commands:
    - train: Run training
    - debug: Run training in debug mode
    - export: Export model from an existing run
    """

    def train(self, *overrides: str) -> None:
        """Run training.

        Args:
            *overrides: Hydra-style config overrides (e.g., trainer.max_epochs=100, run_id=my-custom-id)

        Examples:
            python run.py train
            python run.py train run_id=my-experiment
            python run.py train trainer.max_epochs=100 data.batch_size=64
        """
        self._run_training(debug=False, overrides=list(overrides))

    def debug(self, *overrides: str) -> None:
        """Run debug/overfit-100 sanity test.

        Args:
            *overrides: Hydra-style config overrides (e.g., trainer.max_epochs=10, run_id=debug-test)

        Examples:
            python run.py debug
            python run.py debug run_id=my-debug-test
        """
        self._run_training(debug=True, overrides=list(overrides))

    def export(
        self,
        run_id: str,
        *overrides: str,
    ) -> None:
        """Export model from an existing run.

        Args:
            run_id: ID of the run to export from (e.g., lcseg-20260107-005456-29a2b97)
            *overrides: Hydra-style config overrides (e.g., checkpoint=last)
        """
        from landcover.training.common import (
            create_model,
            export_model,
            resolve_checkpoint_path,
        )

        run_dir = Path("runs") / run_id
        run_paths = get_run_paths(run_dir)

        # Validate run directory exists
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)

        # Setup logging
        setup_run_logging(run_dir)
        logger = get_logger(__name__)

        logger.info("=" * 60)
        logger.info(f"Export Mode - Run ID: {run_id}")
        logger.info("=" * 60)

        # Load config from saved run
        config_path = run_paths["config_yaml"]
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            sys.exit(1)

        saved_cfg = OmegaConf.load(config_path)
        logger.info(f"Loaded config from {config_path}")

        # Apply overrides using Hydra compose and merge with saved config
        override_cfg = _load_config("training", list(overrides))
        cfg = OmegaConf.merge(saved_cfg, override_cfg)
        if overrides:
            logger.info(f"Applied overrides: {list(overrides)}")

        # Resolve checkpoint path (use checkpoint from config, default to 'best')
        checkpoint_type = cfg.export.get("checkpoint", "best")
        try:
            checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint_type)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"Using checkpoint: {checkpoint_path}")

        # Create model (architecture only, weights loaded during export)
        model = create_model(cfg)

        # Export
        export_model(cfg, model, checkpoint_path, run_paths["export"])

        logger.info(f"Export complete. Files saved to: {run_paths['export']}")

    def _run_training(
        self,
        debug: bool,
        overrides: list[str],
    ) -> None:
        """Internal method to run training.

        Args:
            debug: Whether to run in debug mode
            overrides: Hydra-style config overrides
        """
        # Load training config with overrides
        cfg = _load_config("training", overrides)

        # Use run_id from config or generate one
        final_run_id = cfg.get("run_id") or generate_run_id()

        # Get runs_root from config, create run directory
        runs_root = cfg.trainer.get("runs_root", "runs")
        run_dir = create_run_directory(runs_root, final_run_id)
        run_paths = get_run_paths(run_dir)

        # Setup logging to run directory
        setup_run_logging(run_dir)
        logger = get_logger(__name__)

        # Print run info
        logger.info("=" * 60)
        logger.info(f"Run ID: {final_run_id}")
        logger.info(f"Run Directory: {run_dir}")
        logger.info(f"Mode: {'debug' if debug else 'train'}")
        logger.info("=" * 60)

        # Save config and overrides
        _save_config_and_overrides(cfg, run_paths, overrides)
        logger.info(f"Saved config to {run_paths['config_yaml']}")

        if debug:
            from landcover.training import debug_train

            debug_train(cfg, final_run_id, run_dir)
        else:
            from landcover.training import train

            train(cfg, final_run_id, run_dir)


def main() -> None:
    """Main entrypoint."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
