"""Process setup and checkpoint files shared by the training tools."""
import json
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed

from .build import trainable_state


def init_distributed():
    """-> (rank, world size, device) of this process; joins the NCCL group when torchrun starts several."""
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    if world > 1:
        torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=60))
    return rank, world, torch.device('cuda', local)


def seed_all(seed):
    """Seed Python, NumPy and torch (CPU and CUDA); called before the model is built."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    """Write JSON atomically (temporary file, then rename)."""
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def resume(path, model, optimizer):
    """Load the model and optimizer state of a ``last.pt``; -> (epoch, step) to continue from."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    result = model.load_state_dict(saved['model'], strict=False)
    if result.unexpected_keys or not all(n.startswith('encoder.') for n in result.missing_keys):
        raise ValueError(f'{path} does not match the model: {result}')
    optimizer.load_state_dict(saved['optimizer'])
    return saved['epoch'], saved['step']


def save_last(path, model, optimizer, epoch, step):
    """Write ``last.pt`` (trainable weights, optimizer, position in the run) atomically."""
    tmp = Path(str(path)+'.tmp')
    torch.save(dict(model=trainable_state(model), optimizer=optimizer.state_dict(), epoch=epoch, step=step), tmp)
    os.replace(tmp, path)
