"""Hydra configuration utilities."""

import subprocess

from omegaconf import OmegaConf


def get_git_sha() -> str:
    """Get the current git revision hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver("git_sha", lambda: get_git_sha()[:7], replace=True)
