---
trigger: always_on
---

# Local Environment Setup

This document describes how to work with the local development environment, including dependency management, data pipelines, code quality tooling, and secrets handling.

## Dependency Management

The project uses **pixi** for dependency and environment management.

- All Python packages and external binaries are managed via `pixi`.
- Always execute project commands through `pixi` to ensure the correct environment is used.

Examples:

~~~bash
pixi run python script.py
pixi run my-cli-command
~~~

If multiple environments are defined, specify one explicitly:

~~~bash
pixi run -e dev python script.py
~~~

## Data Management

Data versioning and pipelines are managed with **DVC**.

- The DVC pipeline is defined in `dvc.yaml`.
- DVC is installed via `pixi` and is available only in the `dev` pixi environment.
- All data loading, preprocessing, and pipeline steps should be executed through DVC.

Examples:

~~~bash
pixi run -e dev dvc repro
pixi run -e dev dvc status
~~~

## Pre-commit Hooks

Code formatting, linting, and other automated checks are enforced using **pre-commit**.

- Pre-commit is installed via `pixi` and is available only in the `dev` environment.
- Hooks should be run before committing changes to ensure code quality and consistency.

Run all checks:

~~~bash
pixi run -e dev lint-all
~~~

## Secrets and Environment Variables

All credentials and secrets are stored in a local `.env` file.

- The `.env` file is not tracked in version control.
- Never commit secrets or credentials to the repository.

To load environment variables into your current shell session:

~~~bash
set -a
source .env
set +a
~~~
