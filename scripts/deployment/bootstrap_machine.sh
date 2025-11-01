#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/setup_template.yaml"
ENV_NAME="dsp"
DRY_RUN="false"
RUN_SMOKE="true"

print_help() {
    cat <<'EOF'
Usage: ./scripts/deployment/bootstrap_machine.sh [options]

Prepare a fresh workstation or VM to run the STMGT traffic forecasting project.

Options:
  --config PATH     Path to setup template (default: configs/setup_template.yaml)
  --env-name NAME   Conda environment name to sync (default: dsp)
  --dry-run         Print actions without executing them
  --skip-smoke      Skip the final smoke test import check
  --help            Show this help message

Examples:
  ./scripts/deployment/bootstrap_machine.sh
  ./scripts/deployment/bootstrap_machine.sh --dry-run
  ./scripts/deployment/bootstrap_machine.sh --env-name dsp-dev --config custom_setup.yaml
EOF
}

log() {
    printf '[bootstrap] %s\n' "$1"
}

error_exit() {
    printf '[bootstrap][error] %s\n' "$1" >&2
    exit 1
}

maybe_source_conda() {
    if command -v conda >/dev/null 2>&1; then
        return
    fi

    local candidate=""

    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidate="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [[ -d "$HOME/miniconda3" ]]; then
        candidate="$HOME/miniconda3"
    elif [[ -d "$HOME/Miniconda3" ]]; then
        candidate="$HOME/Miniconda3"
    elif [[ -d "$HOME/anaconda3" ]]; then
        candidate="$HOME/anaconda3"
    fi

    if [[ -n "$candidate" && -f "$candidate/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$candidate/etc/profile.d/conda.sh"
    fi

    if ! command -v conda >/dev/null 2>&1; then
        error_exit "Conda command not available. Install Miniconda or ensure conda.sh is sourced."
    fi
}

ensure_config_present() {
    if [[ ! -f "$CONFIG_PATH" ]]; then
        error_exit "Setup template '$CONFIG_PATH' not found."
    fi
    log "Using setup template: $CONFIG_PATH"
}

ensure_env_synced() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry-run: would sync Conda environment '$ENV_NAME' using environment.yml"
        return
    fi

    if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
        log "Updating existing Conda environment '$ENV_NAME'"
        conda env update --name "$ENV_NAME" --file environment.yml --prune
    else
        log "Creating Conda environment '$ENV_NAME'"
        conda env create --name "$ENV_NAME" --file environment.yml
    fi
}

install_project_package() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry-run: would install project in editable mode"
        return
    fi

    log "Installing project package into '$ENV_NAME'"
    conda run -n "$ENV_NAME" pip install -e .
}

prepare_directories() {
    local dirs=(cache data data/processed outputs logs models/training_runs)
    for dir_path in "${dirs[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log "Dry-run: would ensure directory '$dir_path' exists"
        else
            mkdir -p "$dir_path"
        fi
    done
}

ensure_env_file() {
    if [[ -f .env ]]; then
        log "Found existing .env file"
        return
    fi

    if [[ ! -f .env.example ]]; then
        log "No .env or .env.example found; skipping template copy"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry-run: would copy .env.example to .env"
        return
    fi

    log "Copying .env.example to .env"
    cp .env.example .env
    log "Fill in secrets in .env before running data collection"
}

run_smoke_check() {
    if [[ "$RUN_SMOKE" != "true" ]]; then
        log "Skipping smoke test"
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry-run: would execute smoke test in '$ENV_NAME'"
        return
    fi

    log "Running smoke test (imports traffic_forecast)"
    if ! conda run -n "$ENV_NAME" python -c 'import traffic_forecast; print("traffic_forecast ready")'; then
        error_exit "Smoke test failed. Check environment configuration."
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -lt 2 ]] && error_exit "--config requires a path"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --env-name)
            [[ $# -lt 2 ]] && error_exit "--env-name requires a value"
            ENV_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-smoke)
            RUN_SMOKE="false"
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            error_exit "Unknown option '$1'"
            ;;
    esac
done

maybe_source_conda
ensure_config_present
prepare_directories
ensure_env_file
ensure_env_synced
install_project_package
run_smoke_check

log "Bootstrap complete. Review .env and follow docs/README_SETUP.md for next steps."
