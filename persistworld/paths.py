"""Filesystem roots. Every data path in a config is relative to ``DATA_ROOT``.

    PERSISTWORLD_DATA   datasets and pretrained weights (default: <repository>/data)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('PERSISTWORLD_DATA', REPO_ROOT/'data'))
CONFIG_DIR = REPO_ROOT/'configs'


def data_path(value):
    """A config path: absolute paths are kept, relative ones are resolved under DATA_ROOT."""
    path = Path(value)
    return path if path.is_absolute() else DATA_ROOT/path
