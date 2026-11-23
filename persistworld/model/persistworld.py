"""PersistWorld: a persistent world state updated once per capture group.

When a capture group arrives, the evolver carries the state to its capture time, the writer writes
the evidence of the new images into it, and the reader decodes the current occupancy. To forecast,
the evolver advances a copy of the state to each queried time along the ego trajectory and the
reader decodes it in the queried ego frame; forecasts never modify the state.

A scene ``Session`` holds the state (fixed-address memory tiles in the scene frame), the observed
ego poses and times, and the ids of the capture groups already written.
"""
from dataclasses import dataclass, field, replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import ego_state as ego_state_lib
from . import evolver as evolver_lib
from . import geometry
from . import host_sync as hs
from . import reader as reader_lib
from .evolver import Evolver
from .memory import SceneMemory, window_box
from .occupancy_writer import OccupancyWriter
from .reader import Reader
from .state import PackedWorldQuery, SceneField
from .writer import CameraWriter

# Seeds of the writer and of the evolver/reader initialization (independent of the global RNG).
WRITER_SEED = 4101
EVOLVER_SEED = 7101


@dataclass
class PersistWorldConfig:
    dim: int = 384                          # packed cell feature (8 sub-cells x 48)
    heads: int = 6
    evolver_layers: int = 3                 # residual blocks of the evolver
    context_layers: int = 3                 # axial blocks of the observed context
    writer_layers: int = 2
    state_shape: tuple = (50, 50, 4)        # cells of 1.6 m covering the occupancy volume
    occupancy_shape: tuple = (200, 200, 16)
    bounds: tuple = (-40., -40., -1., 40., 40., 5.4)
    groups_per_write: int = 4               # windowed mode: capture groups written together
    num_classes: int = 18
    free_id: int = 17
    tile_shape: tuple = (10, 10, 4)         # memory tile, in cells
    halo: int = 1
    depth_bins: int = 75                    # ray-depth sections of 0.8 m
    depth_bin_size: float = .8


@dataclass
class Session:
    """Streaming state of one scene: the scene frame, the memory, the observed poses/times and the groups written."""
    scene: str
    world_from_scene: torch.Tensor
    memory: SceneMemory
    poses: torch.Tensor
    times: torch.Tensor
    last_time: float
    updates: int = 0
    consumed: set = field(default_factory=set)

    def fork(self):
        """Independent copy: tensor storage is shared until a write, all containers are private."""
        return Session(self.scene, self.world_from_scene, self.memory.fork(), self.poses, self.times,
                       self.last_time, self.updates, self.consumed.copy())


class PersistWorld(nn.Module):
    """World state with its writer (camera or occupancy input), evolver and reader."""
    def __init__(self, encoder, config=None, writer='camera'):
        super().__init__()
        self.config = c = config or PersistWorldConfig()
        if writer not in ('camera', 'occupancy'):
            raise ValueError(writer)
        self.encoder = encoder
        self.initial = nn.Sequential(nn.Linear(3, c.dim), nn.GELU(), nn.Linear(c.dim, c.dim))
        self.cell_size = tuple((c.bounds[i+3]-c.bounds[i])/c.state_shape[i] for i in range(3))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(EVOLVER_SEED)
            evolver = Evolver(c.dim, c.heads, c.evolver_layers, c.context_layers)
            reader = Reader(c.dim, c.num_classes)
        if writer == 'camera':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(WRITER_SEED)
                writer = CameraWriter(encoder.feature_dim, c.dim, self.cell_size, layers=c.writer_layers,
                                      depth_bins=c.depth_bins, depth_bin_size=c.depth_bin_size)
        else:
            writer = OccupancyWriter(c.dim, self.cell_size, c.num_classes, c.bounds, c.occupancy_shape,
                                     c.writer_layers)
        self.evolver = evolver
        self.writer = writer
        self.reader = reader
        self.checkpoint_readout = False      # execution option (execution.configure_execution)
        self.ablate_evolver = False          # ablation abl_no_evolver: the state is read without advancing it
        self.ablate_shared_context = False   # ablation abl_no_context
        self.ego_state = False               # ablation abl_ego_state: the lattice follows the ego

    # ------------------------------------------------------------------ sessions and state windows

    def new_session(self, scene, world_from_ego, time):
        """Empty state of ``scene``; its fixed frame is the ego frame of the first capture group."""
        if world_from_ego.shape != (1, 4, 4):
            raise ValueError('One scene per session')
        c = self.config
        return Session(scene, hs.view(world_from_ego, lambda x: x[0].detach().clone()),
                       SceneMemory(c.dim, self.cell_size, c.tile_shape),
                       hs.view(world_from_ego, lambda x: x[:, None, :].detach()),
                       hs.view(time, lambda x: x[:, None].detach()), float(hs.host(time)[0]))

    @staticmethod
    def observation_stamp(session):
        """Identifies the observed state of a session (an observed context is valid only for its stamp)."""
        return (session.scene, session.updates, session.last_time, session.memory.revision)

    def _prepare_forecast_context(self, session):
        """The evolver's shared context of the observed state (see evolver.prepare_context)."""
        return evolver_lib.prepare_context(self, session)

    def _prediction_inputs(self, session, poses):
        """State window around the ego at the last pose of ``poses`` and its ego-relative cell positions."""
        pose = hs.view(poses, lambda x: x[:, -1])
        scene_from_ego = geometry.solve(session.world_from_scene, pose[0])
        # The integer window comes from host geometry (CPU mirrors) when sync-free.
        box_from = (geometry.solve(hs.host(session.world_from_scene), hs.host(pose)[0]) if hs.ENABLED
                    else scene_from_ego)
        origin, shape = window_box(box_from, self.config.bounds, self.cell_size, self.config.halo)
        axes = [(torch.arange(n, device=pose.device).float()+o+.5)*s
                for n, o, s in zip(shape, origin, self.cell_size)]
        points = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
        # Network positions are relative to the query ego frame; stored addresses stay fixed.
        ego_from_scene = geometry.inverse(scene_from_ego)
        ego_points = geometry.transform_points(ego_from_scene, points.reshape(-1, 3)).reshape_as(points)
        low = hs.const(list(self.config.bounds[:3]), points)
        span = hs.const(list(self.config.bounds[3:]), points)-low
        normalized = 2*(ego_points-low)/span-1
        initial = self.initial(normalized)[None]
        window = session.memory.read(origin, initial, session.last_time)
        return window, normalized

    def _predict(self, session, poses, times, context=None):
        """The state window advanced to the last of ``times`` (the evolver; identity at zero elapsed time)."""
        window, normalized = self._prediction_inputs(session, poses)
        target = times[:, -1]
        if self.ablate_evolver:
            return window.replace(window.features, state_time=target[:, None, None, None].expand_as(window.state_time))
        if context is None:
            context = self._prepare_forecast_context(session)
        if hs.checks() and torch.any(target < context.reference_time-1e-5):
            raise ValueError('Target precedes the observed context')
        delta = context.crop(self, session, window)
        if self.ablate_shared_context:
            delta = torch.zeros_like(delta)
        return self.evolver(window, poses, times, target, normalized, context_delta=delta)

    def _as_ego(self, session, window, pose, time):
        """Reader request for ``window`` seen from the ego ``pose`` at ``time``."""
        return PackedWorldQuery(window.features, pose, time, hs.upload([session.last_time], time), session.updates,
                                origin=window.origin, cell_size=window.cell_size,
                                scene_from_ego=geometry.solve(session.world_from_scene, pose[0]))

    def _occupancy_readout(self, ego):
        """Occupancy logits [1,X,Y,Z,classes] of a reader request on the ego grid."""
        def read(features):
            return reader_lib.read_world_query(self.reader, replace(ego, features=features),
                                               self.config.occupancy_shape, self.config.bounds)
        if self.training and self.checkpoint_readout:
            return checkpoint(read, ego.features, use_reentrant=False)
        return read(ego.features)

    # ------------------------------------------------------------------ write, read, forecast

    def observe(self, batch, session=None, encoded=None, current_output=None):
        """Advance the state to the capture time of ``batch`` and write its capture groups not yet written.

        ``encoded``: the encoder output of the groups when it was computed outside (``encoder.encode``).
        ``current_output``: a list that receives the current-frame readout {'occupancy', 'state'}.
        """
        scene = batch['scene']
        rgb = batch['rgb']
        times = batch['times']
        if rgb.shape[0] != 1 or times.shape[0] != 1:
            raise ValueError('One scene per stream')
        if session is None or scene != session.scene:
            session = self.new_session(scene, hs.view(batch['world_from_ego'], lambda x: x[:, 0]),
                                       hs.view(times, lambda x: x[:, 0]))
        if hs.checks() and torch.any(times[:, 1:] < times[:, :-1]):
            raise ValueError('Capture times must be ordered')
        frame_ids = batch['frame_ids']
        if len(frame_ids) != times.shape[1]:
            raise ValueError('Missing frame identifiers')
        keep = [i for i, ids in enumerate(frame_ids) if tuple(ids) not in session.consumed]
        if not keep:
            if current_output is not None:
                raise ValueError('Current readout requires a new observation')
            return session
        times = hs.take(times, 1, keep)
        first = float(hs.host(times)[0, 0])
        if session.updates and first < session.last_time:
            raise ValueError('Unseen images arrived out of order; cannot rewind a scene')
        rgb = hs.take(rgb, 1, keep)
        poses = hs.take(batch['world_from_ego'], 1, keep)
        projections = hs.take(batch['image_from_world'], 1, keep)
        image_from_scene = geometry.compose(projections, session.world_from_scene[None, None, None])
        if session.updates:
            all_poses = hs.cat([session.poses, poses], 1)
            all_times = hs.cat([session.times, times], 1)
            offset = session.times.shape[1]
        else:
            all_poses = poses
            all_times = times
            offset = 0
        # The evolver carries the state to the capture time of the last group written here.
        last = offset+times.shape[1]
        end = self._predict(session, hs.view(all_poses, lambda x: x[:, :last]),
                            hs.view(all_times, lambda x: x[:, :last]))
        if encoded is not None and keep != list(range(len(frame_ids))):
            raise ValueError('Precomputed features no longer match the groups to write')
        evidence = self.encoder(rgb) if encoded is None else self.encoder.writer_input(encoded)
        prior = SceneField(end.features, poses[:, -1], times[:, -1],
                           hs.upload([session.last_time if session.updates else -torch.inf], times), session.updates)
        posterior, coverage = self.writer(prior, evidence['features'], image_from_scene, times, rgb.shape[-2:],
                                          evidence['patch_grid'], end.points())
        observed = torch.where(coverage, times[:, -1, None, None, None], end.observed_time)
        end = end.replace(posterior.features, observed_time=observed)
        if current_output is not None:
            ego = self._as_ego(session, end, batch.get('current_world_from_ego', poses[:, -1]), times[:, -1])
            current_output.append({'occupancy': self._occupancy_readout(ego), 'state': ego})
        session.memory.write(end)
        if self.ego_state:
            ego_state_lib.recenter(self, session, hs.view(poses, lambda x: x[:, -1]), end.features.dtype)
        session.poses = hs.view(all_poses, lambda x: x.detach())
        session.times = hs.view(all_times, lambda x: x.detach())
        session.last_time = float(hs.host(times)[0, -1])
        session.updates += 1
        session.consumed.update(tuple(frame_ids[i]) for i in keep)
        return session

    def forecast_stream(self, session, batch):
        """Advance copies of the state to every future pose/time of ``batch`` and read them out."""
        result = []
        poses = batch['future_world_from_ego']
        times = batch['future_times']
        if times.shape[1] and float(hs.host(times)[0, 0]) <= session.last_time:
            raise ValueError('Forecast must follow the latest observation')
        context = self._prepare_forecast_context(session) if times.shape[1] else None
        for i in range(times.shape[1]):
            history = hs.cat([session.poses, hs.view(poses, lambda x: x[:, :i+1])], 1)
            stamps = hs.cat([session.times, hs.view(times, lambda x: x[:, :i+1])], 1)
            predicted = self._predict(session, history, stamps, context=context)
            ego = self._as_ego(session, predicted, poses[:, i], times[:, i])
            result.append({'occupancy': self._occupancy_readout(ego), 'state': ego})
        return result

    def forward_query(self, batch, current_readout=False):
        """Windowed mode: rebuild a fresh state from the query's input window (blocks of ``groups_per_write``
        capture groups, repeated groups written once) and forecast. Returns the forecast items and,
        with ``current_readout``, the current-frame readout."""
        if batch.get('history_mode') != 'window8':
            raise ValueError('Windowed queries carry an eight-group input window')
        current = [] if current_readout else None
        pieces = window_blocks(batch, self.config.groups_per_write)
        session = None
        for index, piece in enumerate(pieces):
            final = index == len(pieces)-1
            session = self.observe(piece, session, current_output=current if final else None)
        if session is None:
            raise ValueError('Empty input window')
        result = self.forecast_stream(session, batch)
        if current_readout:
            if len(current) != 1:
                raise ValueError('Current readout missing')
            return result, current[0]
        return result


def window_blocks(batch, groups_per_write):
    """Split an input window into chronological blocks of ``groups_per_write`` groups; repeated groups are
    written once."""
    n = len(batch['frame_ids'])
    if batch['rgb'].shape[1] != n or n % groups_per_write:
        raise ValueError('Input windows need whole blocks of groups, including padding')
    times = hs.host(batch['times'])
    if torch.any(times[:, 1:] < times[:, :-1]):
        raise ValueError('Input groups must be chronological')
    seen = set()
    pieces = []
    for start in range(0, n, groups_per_write):
        keep = []
        for i in range(start, min(start+groups_per_write, n)):
            group = tuple(batch['frame_ids'][i])
            if group not in seen:
                keep.append(i)
                seen.add(group)
        if not keep:
            continue
        piece = {**batch, 'frame_ids': [batch['frame_ids'][i] for i in keep]}
        for key in ('rgb', 'times', 'world_from_ego', 'image_from_world'):
            piece[key] = hs.take(batch[key], 1, keep)
        pieces.append(piece)
    return pieces
