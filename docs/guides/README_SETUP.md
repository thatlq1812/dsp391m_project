# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Cross-Machine Setup Guide

Guidance for preparing a fresh workstation or cloud VM to run the STMGT traffic forecasting project.

## Quick Checklist

- Clone the repository and pull required artifacts.
- Review `configs/setup_template.yaml` and adjust values if your environment differs.
- Create a Conda environment with `environment.yml` and install the project in editable mode.
- Copy `.env.example` to `.env`, then provide API keys and deployment-specific paths.
- Validate the installation with the bootstrap script and a smoke test.

## 1. Clone the Repository

```bash
# Using Git Bash / VS Code terminal
cd /path/to/projects
git clone git@github.com:thatlq1812/dsp391m_project.git
cd dsp391m_project
```

If the repository was transferred manually, verify that `cache/`, `data/`, and `outputs/` contain the expected sub-directories. Create empty folders when necessary (see template).

## 2. Review the Setup Template

`configs/setup_template.yaml` centralizes machine-specific expectations:

- Environment metadata (`conda_env`, Python version, default ports).
- Required directories for data, cache, models, and logs.
- Lists of mandatory environment variables and smoke-test commands.
- Optional remote deployment placeholders (host, username, mount points).

Adjust the template or create a copy if a machine needs overrides (for example, distinct Conda env names).

## 3. Bootstrap the Environment

The helper script installs dependencies, ensures directories exist, and performs a smoke test.

```bash
./scripts/deployment/bootstrap_machine.sh
```

Key options:

- `--dry-run` prints actions without executing them.
- `--env-name dsp-dev` targets an alternate Conda environment.
- `--skip-smoke` skips the final import test when Python is not yet fully configured.

The script automatically copies `.env.example` to `.env` on first run so you can edit credentials safely.

## 4. Provide Secrets and Runtime Configuration

1. Open `.env` and update placeholders (`GOOGLE_MAPS_API_KEY`, optional URLs, and runtime toggles).
2. Confirm ports (`STREAMLIT_SERVER_PORT`, `FASTAPI_SERVER_PORT`) are available on the target machine.
3. Set `PROJECT_ENV` to `production` for deployed instances or `development` for local experimentation.
4. Provide `CONDA_FALLBACK_PATH` when Conda is installed in a non-standard location so the dashboard tooling can invoke `conda run` reliably.
5. Keep `.env` out of version control (`.gitignore` already handles this).

## 5. Validate the Installation

Run quick diagnostics after environment bootstrap:

```bash
# Basic import check
conda run -n dsp python -c "import traffic_forecast; print('ready')"

# Optional smoke tests
conda run -n dsp pytest -k smoke --maxfail=1

# Launch the dashboard (headless optional)
./run_dashboard.sh --port 8505 --headless
```

If any command fails, revisit the bootstrap script output and verify environment variables defined in `.env` and `configs/setup_template.yaml`.

## 6. Remote Deployment Notes

- Populate the `remote` section of `configs/setup_template.yaml` before using `scripts/deployment/deploy_git.sh` or VM automation scripts.
- Ensure SSH key-based access is configured; the deployment scripts assume non-interactive authentication.
- Synchronize cache and data directories manually or via existing utility scripts in `scripts/data/` when migrating historical artifacts.

## 7. Recommended Follow-Up

- Update `configs/project_config.yaml` only after confirming the target machine matches the expected data paths and time zone.
- Document machine-specific deviations (custom ports, schedulers) inside a copy of the setup template for troubleshooting.
- Re-run `bootstrap_machine.sh` with `--dry-run` after major dependency updates to understand upcoming changes before applying them.
