"""Utility helpers for resolving Conda executables cross-platform."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETUP_TEMPLATE_PATH = PROJECT_ROOT / "configs" / "setup_template.yaml"


def _expand_path(path_value: str) -> Path:
    """Normalize a path string with environment and user expansion."""
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    return Path(expanded)


def _iter_configured_paths() -> Iterable[Path]:
    """Yield fallback Conda paths declared in the setup template."""
    if not SETUP_TEMPLATE_PATH.exists() or yaml is None:
        return []
    try:
        with SETUP_TEMPLATE_PATH.open('r', encoding='utf-8') as handle:
            payload = yaml.safe_load(handle) or {}
    except (yaml.YAMLError, OSError, AttributeError):  # type: ignore[union-attr]
        return []

    environment_config = payload.get('environment', {}) if isinstance(payload, dict) else {}
    fallback_values = environment_config.get('conda_fallback_paths', [])
    if isinstance(fallback_values, str):
        fallback_values = [fallback_values]

    paths: List[Path] = []
    for value in fallback_values or []:
        if not value:
            continue
        try:
            paths.append(_expand_path(str(value)))
        except TypeError:
            continue
    return paths


@lru_cache(maxsize=1)
def _conda_candidates() -> List[Path]:
    """Collect candidate Conda executables in priority order."""
    candidates: List[Path] = []

    env_exe = os.environ.get('CONDA_EXE')
    if env_exe:
        candidates.append(_expand_path(env_exe))

    fallback_env = os.environ.get('CONDA_FALLBACK_PATH')
    if fallback_env:
        candidates.append(_expand_path(fallback_env))

    candidates.extend(_iter_configured_paths())

    default_paths = [
        Path('C:/ProgramData/miniconda3/Scripts/conda.exe'),
        Path('C:/Users/Public/miniconda3/Scripts/conda.exe'),
        Path('C:/miniconda3/Scripts/conda.exe'),
        Path.home() / 'miniconda3' / 'bin' / 'conda',
        Path.home() / 'anaconda3' / 'bin' / 'conda',
        Path('/usr/local/bin/conda'),
        Path('/opt/conda/bin/conda'),
    ]
    candidates.extend(default_paths)

    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        path_str = str(candidate)
        if path_str not in seen:
            seen.add(path_str)
            unique.append(candidate)
    return unique


def resolve_conda_executable() -> str:
    """Return the first available Conda executable or fallback to 'conda'."""
    for candidate in _conda_candidates():
        if candidate.exists():
            return str(candidate)
    return 'conda'
