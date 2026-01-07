"""Tile catalog for discovering and accessing inference tiles."""

from pathlib import Path

import rasterio
from omegaconf import DictConfig

from utils.logging import get_logger

logger = get_logger(__name__)


class TileCatalog:
    """Catalog of available inference tiles organized by country."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize catalog by scanning data directories.

        Args:
            cfg: Hydra configuration with data settings
        """
        self.imagery_root = Path(cfg.data.imagery_root)
        self.labels_root = Path(cfg.data.labels_root)
        self.tile_glob = cfg.data.tile_glob
        self.allowed_countries = cfg.data.allowed_countries

        # Build catalog: {country: {tile_id: {"imagery": path, "label": path}}}
        self._catalog: dict[str, dict[str, dict[str, Path]]] = {}
        self._build_catalog()

    def _build_catalog(self) -> None:
        """Scan directories and build tile catalog.

        Handles structure: imagery_root/<country>/<tile_hash>/spectral.tif
        """
        logger.info(f"Building tile catalog from {self.imagery_root}")

        if not self.imagery_root.exists():
            logger.warning(f"Imagery root not found: {self.imagery_root}")
            return

        for country_dir in self.imagery_root.iterdir():
            if not country_dir.is_dir():
                continue

            country = country_dir.name

            # Skip hidden directories
            if country.startswith("."):
                continue

            # Apply country filter if set
            if self.allowed_countries and country not in self.allowed_countries:
                continue

            self._catalog[country] = {}

            # Scan for tile directories (hash-named subdirs)
            for tile_dir in country_dir.iterdir():
                if not tile_dir.is_dir():
                    continue

                # Skip hidden directories
                if tile_dir.name.startswith("."):
                    continue

                tile_id = tile_dir.name

                # Look for spectral.tif inside the tile directory
                spectral_path = tile_dir / "spectral.tif"
                if not spectral_path.exists():
                    # Fallback: try glob pattern for any .tif
                    tif_files = list(tile_dir.glob(self.tile_glob))
                    if tif_files:
                        spectral_path = tif_files[0]
                    else:
                        continue  # No imagery found

                # Look for corresponding label in same structure
                label_path = self.labels_root / country / tile_id / "labels.tif"
                if not label_path.exists():
                    # Try alternative names
                    alt_label = self.labels_root / country / tile_id / "label.tif"
                    if alt_label.exists():
                        label_path = alt_label
                    else:
                        label_path = None

                self._catalog[country][tile_id] = {
                    "imagery": spectral_path,
                    "label": label_path,
                }

        # Log summary
        total_tiles = sum(len(tiles) for tiles in self._catalog.values())
        logger.info(
            f"Catalog built: {len(self._catalog)} countries, {total_tiles} tiles"
        )

    def get_countries(self) -> list[str]:
        """Get list of available country codes.

        Returns:
            Sorted list of country codes
        """
        return sorted(self._catalog.keys())

    def get_tiles(self, country: str) -> list[str]:
        """Get list of tile IDs for a country.

        Args:
            country: Country code

        Returns:
            Sorted list of tile IDs

        Raises:
            KeyError: If country not found
        """
        if country not in self._catalog:
            raise KeyError(f"Country not found: {country}")
        return sorted(self._catalog[country].keys())

    def get_tile_paths(self, country: str, tile_id: str) -> dict[str, Path | None]:
        """Get paths for a specific tile.

        Args:
            country: Country code
            tile_id: Tile identifier

        Returns:
            Dict with 'imagery' and 'label' paths (label may be None)

        Raises:
            KeyError: If country or tile not found
        """
        if country not in self._catalog:
            raise KeyError(f"Country not found: {country}")
        if tile_id not in self._catalog[country]:
            raise KeyError(f"Tile not found: {country}/{tile_id}")
        return self._catalog[country][tile_id]

    def get_tile_meta(self, country: str, tile_id: str) -> dict:
        """Get metadata for a specific tile using rasterio.

        Args:
            country: Country code
            tile_id: Tile identifier

        Returns:
            Dict with tile metadata (width, height, crs, bounds, band_count)

        Raises:
            KeyError: If tile not found
        """
        paths = self.get_tile_paths(country, tile_id)
        imagery_path = paths["imagery"]

        with rasterio.open(imagery_path) as src:
            return {
                "tile_id": tile_id,
                "country": country,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "bounds": list(src.bounds),
                "band_count": src.count,
            }

    def has_label(self, country: str, tile_id: str) -> bool:
        """Check if a tile has a corresponding label file.

        Args:
            country: Country code
            tile_id: Tile identifier

        Returns:
            True if label file exists
        """
        paths = self.get_tile_paths(country, tile_id)
        return paths["label"] is not None

    @property
    def num_countries(self) -> int:
        """Number of countries in catalog."""
        return len(self._catalog)

    @property
    def num_tiles(self) -> int:
        """Total number of tiles across all countries."""
        return sum(len(tiles) for tiles in self._catalog.values())
