"""CPU smoke test without data or pretrained weights: a few streaming steps, the loss and its backward,
for the camera model (a random stand-in for the image encoder) and the occupancy-input model, on a
reduced volume.

    python tests/test_cpu_smoke.py      (or: pytest tests)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn

from persistworld import streaming as st
from persistworld.config import load_config
from persistworld.losses import frame_loss, objective_spec
from persistworld.model import PersistWorld, PersistWorldConfig
from persistworld.model.execution import configure_execution
from persistworld.model.occupancy_writer import OccupancyInput

H, W = 64, 176
SMALL = dict(state_shape=(20, 20, 4), occupancy_shape=(80, 80, 16), bounds=(-16, -16, -1, 16, 16, 5.4))


class RandomEncoder(nn.Module):
    """Stand-in for DINOv3: a linear patch embedding with the same output interface."""
    feature_dim = 768

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3*16*16, self.feature_dim)

    def encode(self, rgb):
        b, t, n, c, h, w = rgb.shape
        x = rgb.float().reshape(b*t*n, c, h//16, 16, w//16, 16).permute(0, 2, 4, 1, 3, 5).reshape(b*t*n, -1, c*256)/255
        return dict(features=self.proj(x).reshape(b, t, n, -1, self.feature_dim), patch_grid=(h//16, w//16))

    def writer_input(self, encoded):
        return dict(features=encoded['features'].float(), patch_grid=encoded['patch_grid'])

    def forward(self, rgb):
        return self.writer_input(self.encode(rgb))


def ego_from_camera(yaw):
    """A camera at (1, 0, 1.5) m in the ego frame looking along ``yaw`` (camera axes: x right, y down, z forward)."""
    r = np.stack([[math.sin(yaw), -math.cos(yaw), 0.], [0., 0., -1.], [math.cos(yaw), math.sin(yaw), 0.]], 1)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [1., 0., 1.5]
    return m


def world_from_ego(k):
    """Ego pose at keyframe k: about 4 m along x and 0.06 rad of yaw per keyframe."""
    c, s = math.cos(.06*k), math.sin(.06*k)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = [4.*k, .3*k, 0.]
    return m


def frame(k, occupancy=False):
    """Streaming input of keyframe k: six random images, or a random label grid with ``occupancy``."""
    g = np.random.default_rng(k)
    pose = world_from_ego(k)
    horizons = [1, 2, 3]
    if occupancy:
        rgb = torch.from_numpy(g.integers(0, 18, (1, 1, 1, 80, 80, 16), dtype=np.uint8))
        proj = torch.from_numpy(np.linalg.inv(pose)).float()[None, None, None]
        ids = [(f'f{k}',)]
    else:
        k3 = np.array([[60., 0, W/2], [0, 60., H/2], [0, 0, 1.]])
        p = np.eye(4)
        p[:3, :3] = k3
        cams = [ego_from_camera(math.radians(a)) for a in (0, -55, 55, 180, 110, -110)]
        proj = torch.from_numpy(np.stack([p@np.linalg.inv(pose@c) for c in cams])).float()[None, None]
        rgb = torch.from_numpy(g.integers(0, 256, (1, 1, 6, 3, H, W), dtype=np.uint8))
        ids = [tuple(f'f{k}c{i}' for i in range(6))]
    future = np.stack([world_from_ego(k+2*h) for h in horizons])
    return dict(scene='s', anchor_index=k, token=f'f{k}', horizons=horizons, frame_ids=ids, history_mode='stream',
                rgb=rgb, times=torch.tensor([[.5*k]]), world_from_ego=torch.from_numpy(pose).float()[None, None],
                image_from_world=proj, current_world_from_ego=torch.from_numpy(pose).float()[None],
                future_world_from_ego=torch.from_numpy(future).float()[None],
                future_times=torch.tensor([[.5*(k+2*h) for h in horizons]]))


def supervision(n=3):
    """Random labels and masks for the current frame and ``n`` horizons."""
    g = torch.Generator().manual_seed(0)
    labels = lambda m: torch.randint(0, 18, (m, 80, 80, 16), generator=g)
    masks = lambda m: torch.rand((m, 80, 80, 16), generator=g) < .8
    return dict(current_labels=labels(1), current_mask_camera=masks(1), labels=labels(n)[None],
                mask_camera=masks(n)[None])


def run(occupancy):
    """Three streaming keyframes on a reduced volume, the loss of the last one and its backward."""
    cfg = load_config('gt' if occupancy else 'main')
    torch.manual_seed(0)
    encoder = OccupancyInput(18) if occupancy else RandomEncoder()
    model = PersistWorld(encoder, PersistWorldConfig(**{**cfg['model'], **SMALL}),
                         writer='occupancy' if occupancy else 'camera')
    configure_execution(model, dict(sync_free_forward=False))
    o = cfg['objective']
    model.occupancy_objective = objective_spec(o['class_priors'], o['free_weight'], o['loss_scale'],
                                               o['current_weight'])
    model.train()
    stream = st.Stream()
    for k in range(3):
        b = frame(k, occupancy)
        grad = k == 2
        model.writer.record_depth = grad
        model.writer.depth_records.clear()
        with torch.set_grad_enabled(grad):
            cur = stream.observe(model, b)
            fut = stream.forecast(model, b)
            stream.finish()
    loss, ray = frame_loss(model, b, supervision(), fut, cur, lambda: model.writer.depth_records[-1][0, 0])
    loss.backward()
    grads = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())
    assert torch.isfinite(loss) and grads > 0
    assert cur['occupancy'].shape == (1, 80, 80, 16, 18)
    print('occupancy input' if occupancy else 'camera input', 'loss %.4f' % loss.item(),
          'parameters with gradient', grads)


def test_camera_input():
    run(False)


def test_occupancy_input():
    run(True)


if __name__ == '__main__':
    test_camera_input()
    test_occupancy_input()
    print('OK')
