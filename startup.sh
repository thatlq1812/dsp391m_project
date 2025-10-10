#!/bin/bash
set -euo pipefail
LOGFILE="/var/log/collector-startup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Startup script started at $(date)"

# noninteractive apt
export DEBIAN_FRONTEND=noninteractive

# update & basic packages
apt-get update -y
apt-get install -y git wget bzip2 build-essential ca-certificates curl

# Install Miniconda (if not present)
CONDA_DIR="/opt/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "Installing Miniconda..."
  TMP_INSTALL="/tmp/miniconda.sh"
  wget -qO "$TMP_INSTALL" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMP_INSTALL" -b -p "$CONDA_DIR"
  rm -f "$TMP_INSTALL"
  chmod -R a+rx "$CONDA_DIR"
fi

# Initialize conda for bash
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh" || true

# Work dir
WORKDIR="/opt/dsp_project"
if [ ! -d "$WORKDIR" ]; then
  echo "Cloning repository..."
  git clone https://github.com/thatlq1812/dsp391m_project.git "$WORKDIR"
else
  echo "Repository already present, pulling latest..."
  cd "$WORKDIR"
  git pull --rebase || true
fi

cd "$WORKDIR"

# Create conda env if needed (env name: dsp)
if ! conda env list | grep -q "^dsp\s"; then
  if [ -f environment.yml ]; then
    echo "Creating conda env from environment.yml..."
    conda env create -f environment.yml -n dsp || conda env update -f environment.yml -n dsp
  else
    echo "No environment.yml found, installing via pip requirements.txt..."
    conda create -y -n dsp python=3.10
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate dsp
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
    conda deactivate
  fi
else
  echo "Conda env 'dsp' already exists."
fi

# Ensure scripts are executable
chmod +x scripts/run_interval.sh scripts/collect_and_render.py || true

# Run the data collection loop for 6 hours (interval 900s = 15min)
echo "Starting collection for 6 hours with 15-minute interval..."
source "$CONDA_DIR/etc/profile.d/conda.sh"
# Use timeout to stop after 6h; the run_interval.sh accepts interval in seconds.
# Adjust command if you prefer to call collect_and_render.py directly.
timeout 6h bash -lc "conda activate dsp && bash scripts/run_interval.sh 900 --no-visualize"

# Alternatively, if run_interval.sh doesn't accept --no-visualize, run:
# timeout 6h bash -lc "conda activate dsp && python scripts/collect_and_render.py --interval 900 --no-visualize"

echo "Collection finished at $(date). Shutting down VM."
# Sync disks then shut down to preserve results
sync
sudo shutdown -h now