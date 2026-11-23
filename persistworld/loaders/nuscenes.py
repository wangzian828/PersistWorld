"""nuScenes metadata tables (camera groups, poses, sweeps). No LiDAR data is read."""
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def quaternion_matrix(q):
    """Rotation matrix of a (w, x, y, z) quaternion."""
    q = np.asarray(q, dtype=np.float64)
    q = q/np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(record):
    """4x4 transform of a nuScenes record with ``rotation`` and ``translation``."""
    m = np.eye(4)
    m[:3, :3] = quaternion_matrix(record['rotation'])
    m[:3, 3] = record['translation']
    return m


def select_history(groups, training=False, rng=None, count=7):
    """SparseWorld-TC's history sampling from the newest-first sweep groups: validation takes every
    sixth group (the previous keyframes); training draws one interval in [4, 8] per query."""
    if not groups:
        return []
    if not training:
        indices = [(i+1)*6-1 for i in range(count)]
    elif len(groups) <= count:
        indices = list(range(len(groups)))+[len(groups)-1]*(count-len(groups))
    else:
        rng = np.random if rng is None else rng
        maximum = min(len(groups)//count, 8)
        minimum = min(maximum, 4)
        draw = rng.integers if hasattr(rng, 'integers') else rng.randint
        interval = int(draw(minimum, maximum+1))
        indices = [(i+1)*interval-1 for i in range(count)]
    return [groups[min(i, len(groups)-1)] for i in indices]


def chronological_groups(groups, sample_data):
    """Reorder groups given as (current, then history from near to far), SparseWorld-TC's order, into
    chronological order with the current group last. Returns (groups, order); no id changes."""
    n = len(groups)
    if n < 2 or n % 4 or any(len(g) != 6 for g in groups):
        raise ValueError('Expected a multiple of four groups of six cameras')
    current = groups[0]
    for group in groups:
        if any(sample_data[x]['timestamp'] > sample_data[c]['timestamp'] for x, c in zip(group, current)):
            raise ValueError('History contains a capture after the current camera')
    order = sorted(list(range(n-1, 0, -1))+[0], key=lambda i: sample_data[groups[i][0]]['timestamp'])
    assert order[-1] == 0
    return [groups[i] for i in order], order


class NuScenesTables:
    """The nuScenes metadata tables needed for camera inputs and label poses, indexed by token."""
    def __init__(self, metadata_root):
        root = Path(metadata_root)
        for name in ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'scene'):
            records = json.loads((root/(name+'.json')).read_text())
            setattr(self, name, {r['token']: r for r in records})
        self.sample_cameras = {}
        self.sample_reference = {}
        for sd in self.sample_data.values():
            if sd['is_key_frame'] and Path(sd['filename']).parts[-2] == 'LIDAR_TOP':
                self.sample_reference[sd['sample_token']] = sd['token']
            if sd['is_key_frame'] and sd['fileformat'] in ('jpg', 'png'):
                channel = Path(sd['filename']).parts[-2]
                if channel in CAMERAS:
                    self.sample_cameras.setdefault(sd['sample_token'], {})[channel] = sd['token']

    def group(self, sample_token):
        """sample_data tokens of the six keyframe images of a sample, in CAMERAS order."""
        return [self.sample_cameras[sample_token][c] for c in CAMERAS]

    def reference_pose(self, sample_token):
        """Ego pose of the sample's reference frame (the frame of the Occ3D labels); metadata only."""
        sd = self.sample_data[self.sample_reference[sample_token]]
        return pose(self.ego_pose[sd['ego_pose_token']])

    def past_groups(self, token, limit=150):
        """Newest-first camera groups before ``token``: sweeps and keyframes."""
        groups = []
        sample = self.sample[token]
        while len(groups) < limit and sample['prev']:
            current = self.group(sample['token'])
            for _ in range(5):
                previous = [self.sample_data[x]['prev'] for x in current]
                if not all(previous):
                    break
                current = previous
                groups.append(current)
            sample = self.sample[sample['prev']]
            groups.append(self.group(sample['token']))
        return groups

    def camera_record(self, token):
        """(sample_data, intrinsics K, world_from_ego, ego_from_camera, image_from_world) of one image."""
        sd = self.sample_data[token]
        cal = self.calibrated_sensor[sd['calibrated_sensor_token']]
        world_from_ego = pose(self.ego_pose[sd['ego_pose_token']])
        ego_from_camera = pose(cal)
        k = np.asarray(cal['camera_intrinsic'])
        projection = np.eye(4)
        projection[:3, :3] = k
        return sd, k, world_from_ego, ego_from_camera, projection@np.linalg.inv(world_from_ego@ego_from_camera)


def size_bytes(value):
    """Total byte size of the arrays and tensors in a (nested) value."""
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(size_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(size_bytes(v) for v in value)
    return 0


class ByteLRU:
    """LRU cache bounded by the byte size of its values."""
    def __init__(self, budget):
        self.budget = budget
        self.bytes = 0
        self.items = OrderedDict()
        self.counts = Counter()

    def get(self, key, build):
        """Cached value of ``key``, else ``build()`` (cached when it fits the budget)."""
        if key in self.items:
            self.counts['hits'] += 1
            value = self.items.pop(key)
            self.items[key] = value
            return value
        self.counts['misses'] += 1
        value = build()
        size = size_bytes(value)
        if size <= self.budget:
            while self.items and self.bytes+size > self.budget:
                _, old = self.items.popitem(last=False)
                self.bytes -= size_bytes(old)
            self.items[key] = value
            self.bytes += size
        return value
