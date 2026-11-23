"""Streaming inference: one state per scene, updated once per capture group, reset periodically.

Shared by training and evaluation, so both run the same steps. At every keyframe the evolver
carries the state to the capture time, the writer writes the new capture group (each image is
encoded once), the reader decodes the current occupancy and the forecasts are read from advanced
copies of the state. The state is reset to the initial state at the start of each scene and after
every ``period`` capture groups (8 in the paper), so it holds between one and eight groups.

``Retention`` drops history explicitly after every keyframe: the evolver's pose history keeps the
last ``keep_poses`` groups, and a memory tile not written during the last ``keep_steps`` keyframes
is dropped. Within a reset period of at most ``keep_steps`` groups no tile is dropped.
"""
import torch

from .model import host_sync as hs

KEEP_POSES = 8
KEEP_STEPS = 8
PERIOD = 8
GEOMETRY_KEYS = ('times', 'world_from_ego', 'image_from_world', 'current_world_from_ego', 'future_world_from_ego',
                 'future_times')


def pin(d):
    """Copy of a batch dict with its CPU tensors in page-locked memory (for asynchronous copies)."""
    return {k: (v.pin_memory() if isinstance(v, torch.Tensor) and v.device.type == 'cpu' else v)
            for k, v in d.items()}


def transfer(cpu, device):
    """A CPU frame on ``device``: geometry keeps its CPU mirror (host_sync), images copy asynchronously if pinned."""
    out = dict(cpu)
    for k in GEOMETRY_KEYS:
        out[k] = hs.to_device(cpu[k], device)
    out['rgb'] = cpu['rgb'].to(device, non_blocking=cpu['rgb'].is_pinned())
    return out


class Retention:
    """History a stream keeps: the last ``keep_poses`` pose groups and the tiles written in the last
    ``keep_steps`` keyframes."""
    def __init__(self, keep_poses=KEEP_POSES, keep_steps=KEEP_STEPS):
        if keep_poses < 2 or keep_steps < 1:
            raise ValueError('Invalid retention')
        self.keep_poses = int(keep_poses)
        self.keep_steps = int(keep_steps)


class Stream:
    """The carried state of one scene stream (session and retention bookkeeping)."""
    def __init__(self, retention=None):
        self.retention = retention or Retention()
        self.reset()

    def reset(self):
        self.session = None
        self.scene = None
        self.step = 0
        self.written = {}
        self.ids = []

    def observe(self, model, b, encoded=None, read=True):
        """Advance the state to keyframe ``b``, write its capture group and, if ``read``, return the current readout.

        ``encoded``: the encoder output of this group, when it is computed outside (``encoder.encode``).
        """
        if b['scene'] != self.scene:
            self.reset()
            self.scene = b['scene']
            self.session = model.new_session(b['scene'], hs.view(b['world_from_ego'], lambda x: x[:, 0]),
                                             hs.view(b['times'], lambda x: x[:, 0]))
        before = dict(self.session.memory.tiles)
        current = [] if read else None
        self.session = model.observe(b, self.session, encoded=encoded, current_output=current)
        self._written = [key for key, tile in self.session.memory.tiles.items() if before.get(key) is not tile]
        self.ids.append(tuple(b['frame_ids'][-1]))
        return current[0] if read else None

    def forecast(self, model, b):
        """Forecast items at the future poses/times of ``b`` (the state is not modified)."""
        return model.forecast_stream(self.session, b) if b['future_times'].shape[1] else []

    def finish(self):
        """Retention after the keyframe (its forecasts have already been read)."""
        self.step += 1
        for key in self._written:
            self.written[key] = self.step
        session = self.session
        retention = self.retention
        if session.poses.shape[1] > retention.keep_poses-1:
            session.poses = hs.view(session.poses, lambda x: x[:, -(retention.keep_poses-1):])
            session.times = hs.view(session.times, lambda x: x[:, -(retention.keep_poses-1):])
        if len(self.ids) > retention.keep_poses:
            for ids in self.ids[:-retention.keep_poses]:
                session.consumed.discard(ids)
            self.ids = self.ids[-retention.keep_poses:]
        stale = [key for key, step in self.written.items() if self.step-step >= retention.keep_steps]
        for key in stale:
            session.memory.tiles.pop(key, None)
            self.written.pop(key)
        if stale:
            session.memory.revision += 1

    def snapshot(self):
        """Cheap copy (tiles are copy-on-write) to restart a stream from this point."""
        return (None if self.session is None else self.session.fork(), self.scene, self.step, dict(self.written),
                list(self.ids))

    def restore(self, snap):
        """Continue from a ``snapshot()``."""
        session, self.scene, self.step, written, ids = snap
        self.written = dict(written)
        self.ids = list(ids)
        self.session = None if session is None else session.fork()   # every restore gets its own copy


class SingleState:
    """Evaluation runner: one state, reset at scene starts and after every ``period`` keyframes."""
    def __init__(self, period=PERIOD, encode=None):
        if type(period) is not int or not 1 <= period <= 16:
            raise ValueError('Reset period must be in 1..16 (8 in the paper)')
        self.period = period
        self.encode = encode            # optional fn(model, rgb) -> encoder features (latency benchmark)
        self.stream = self._new_stream()
        self.scene = None
        self.last_index = -1

    def _new_stream(self):
        # Periods <= 8: the default retention. Longer periods keep every tile of the period.
        if self.period > KEEP_STEPS:
            return Stream(Retention(KEEP_POSES, self.period))
        return Stream()

    def step_frame(self, model, batch):
        """Consume the next keyframe of the stream -> (current readout, forecast items)."""
        new_scene = batch['scene'] != self.scene
        if new_scene:
            self.scene = batch['scene']
            self.last_index = -1
        index = batch['anchor_index']
        if index != self.last_index + 1:
            raise ValueError('Streaming evaluation must consume every keyframe in order')
        if new_scene or index % self.period == 0:
            self.stream = self._new_stream()
        encoded = None
        if self.encode is not None:
            encoded = dict(features=self.encode(model, batch['rgb']),
                           patch_grid=(batch['rgb'].shape[-2]//16, batch['rgb'].shape[-1]//16))
        current = self.stream.observe(model, batch, encoded=encoded)
        future = self.stream.forecast(model, batch)
        self.stream.finish()
        depth = self.stream.session.updates
        if depth != index % self.period + 1:
            raise AssertionError('State does not follow the reset period')
        self.last_index = index
        return current, future
