#!/usr/bin/env bash
set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./run_dashboard.sh [--port PORT] [--headless] [--help]

Launch the Streamlit dashboard using the `dsp` Conda environment.

Options:
  --port PORT    Streamlit port (default: 8505)
  --headless     Enable headless mode (useful for remote sessions)
  --help         Show this help message

Examples:
  ./run_dashboard.sh
  ./run_dashboard.sh --port 8600
  ./run_dashboard.sh --headless
EOF
}

PORT=8505
HEADLESS_FLAG="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            if [[ $# -lt 2 ]]; then
                echo "Error: --port requires a value" >&2
                exit 1
            fi
            PORT="$2"
            shift 2
            ;;
        --headless)
            HEADLESS_FLAG="true"
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            print_help >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

COMMAND=(
    conda run -n dsp --no-capture-output \
    streamlit run dashboard/Dashboard.py \
    --server.port "${PORT}"
)

if [[ "$HEADLESS_FLAG" == "true" ]]; then
    COMMAND+=(--server.headless true)
fi

printf 'Launching Streamlit dashboard on port %s (headless=%s)\n' "$PORT" "$HEADLESS_FLAG"
"${COMMAND[@]}"
