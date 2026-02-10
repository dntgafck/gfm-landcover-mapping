# Agent Guide (Repository-Specific)

This repository is a Python 3.11 project for land-cover segmentation with:

- Training/export CLI via `run.py` (Fire + Hydra)
- Inference server via FastAPI + ONNX Runtime (`inference/`)
- Data pipeline via DVC (`dvc.yaml`, `data_preparation/`)
- Dev automation via Pixi (`pixi.toml`) and pre-commit
  (`.pre-commit-config.yaml`)

No Cursor rules found (`.cursorrules` or `.cursor/rules/`). No GitHub Copilot
instructions found (`.github/copilot-instructions.md`).

## Quickstart (Environment)

- Recommended setup: `./setup.sh` (installs Pixi if missing, installs deps)
- Activate: `pixi shell` (runtime) or `pixi shell -e dev` (adds `pytest`,
  `pre-commit`)
- No activation: prefix commands with `pixi run ...`

Notes:

- Reproducibility: use `pixi.lock` (`--frozen` install) on local + remote
  (linux-64).
- Secrets: credentials live in `.env` (git-ignored); load via
  `set -a; source .env; set +a`.
- Large artifacts/data are DVC-managed and ignored (`data/`, `runs/`,
  `outputs/`).

## Build / Lint / Test Commands

No traditional “build” step; use `pixi install -e default --frozen` (packaging
in `pyproject.toml`).

### Lint / Format

Primary (matches CI-style checks):

- `pixi run -e dev lint-all`
  - runs `pre-commit run --all-files`

Before finishing any task with code changes: `pixi run -e dev lint-all` must be
clean.

Targeted run: `pixi run -e dev pre-commit run --files path/to/file.py` (or
`black` / `isort` / `flake8`).

Pre-commit hooks: `black`, `isort`, `flake8` (+ bugbear), `prettier`
(md/yaml/toml/json; width 80, prose wrap)

Flake8 ignores (see `.pre-commit-config.yaml`): `E501`, `E402`, `F401`, `E203`,
`E408`.

### Tests

Run tests: `pixi run -e dev test` or `pixi run -e dev pytest`

Run a single test file:

- `pixi run -e dev pytest tests/index/test_splitting.py`

Run a single test function:

- `pixi run -e dev pytest tests/index/test_splitting.py::test_validate_fractions`

Run tests matching a pattern:

- `pixi run -e dev pytest -k validate_fractions`

## Common Project Commands (Operational)

### Training / Debug / Export / Serve (Fire + Hydra)

Entry point: `run.py`

- Train: `pixi run python run.py train` (overrides:
  `trainer.max_epochs=5 data.batch_size=8`)
- Debug: `pixi run python run.py debug`
- Export: `pixi run python run.py export <run_id>` (override:
  `export.checkpoint=last`)
- Serve:
  `pixi run python run.py serve model.local.onnx_path=runs/<run_id>/export/model.onnx`

### DVC (Data Pipeline)

Pipeline definition: `dvc.yaml`

- Run full pipeline: `dvc repro`
- Run a single stage: `dvc repro load_aoi` (or any stage name in `dvc.yaml`)
- Pull artifacts/data tracked by DVC: `dvc pull`

Data contract and workflow rules:

- Respect the training data contract in `README.md`; never change split
  assignments during training.
- OOD is strictly read-only for final evaluation; do not use it for
  normalization or early stopping.
- Execute data steps through DVC; if data processing logic changes, update/check
  `dvc.yaml`.

## Code Style Guidelines (Repository Conventions)

### Formatting

- Formatting is enforced by pre-commit (`black` for Python; `prettier` for
  md/yaml/toml/json).

### Imports

- Use absolute imports; `isort` enforces ordering (stdlib → third-party →
  local).

### Types

- Target Python 3.11+ typing:
  - Prefer built-in generics: `list[str]`, `dict[str, Any]`.
  - Prefer unions with `|`: `Path | None`.
- Add types at public boundaries; no strict type checker configured.

### Naming

- `snake_case` (modules/functions/vars), `PascalCase` (classes),
  `UPPER_SNAKE_CASE` (constants).

### Configuration (Hydra / OmegaConf)

- Configs live in `configs/` and are composed via Hydra.
- Prefer passing config values through Hydra rather than hardcoding.
- When writing new entrypoints, follow the “compose API” pattern used in
  `run.py`:
  - clear Hydra global state (`GlobalHydra.instance().clear()`)
  - `initialize_config_dir(...)` then `compose(...)`
- If you need git metadata in configs, a resolver exists: `${git_sha:}` (see
  `utils/hydra_config.py`).

### Logging

- Prefer `utils.logging` helpers; call
  `setup_logging()`/`setup_run_logging(...)` in entrypoints.
- Avoid `print()` in library code; use `logger.exception(...)` for unexpected
  failures.

### Error Handling

- Library code: raise specific exceptions (`ValueError`, `FileNotFoundError`,
  `KeyError`).
- CLI entrypoints: handle fatal errors at the boundary and exit non-zero
  (`sys.exit(1)`).
- FastAPI routes: map expected failures to `HTTPException` (400/404/503) and log
  unexpected errors.

### Filesystem / Data Access

- Use `pathlib.Path`; prefer `dvc.api.open(...)` for DVC-tracked artifacts.
- Don’t commit large artifacts (`data/`, `runs/`, `outputs/`, `mlflow/` are
  ignored).

### Tests

- Tests are `pytest`-based (configured in `pyproject.toml` under
  `[tool.pytest.ini_options]`).
- Keep tests deterministic and fast; prefer mocks over network/DVC pulls.
- When adding tests, follow existing naming: `tests/**/test_*.py`, `test_*`
  functions.
