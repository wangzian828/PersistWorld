"""Occ3D-nuScenes anchors, camera inputs and occupancy supervision.

Labels are the official Occ3D grids (18 classes, free = 17, 200x200x16 voxels of 0.4 m, z in
[-1, 5.4] m), scored inside the target ``mask_camera``. Every pose given to the model for the
current frame and the future targets is the label reference ego pose of that sample. Images are
resized and bottom-cropped to 256x704 (SparseWorld-TC's crop). Streaming frames carry the camera
group of their keyframe only. Windowed queries (the windowed model of Tab. 3) carry a camera history
as in SparseWorld-TC: the current group plus seven earlier groups from the sweep list (training
interval drawn from [4, 8], validation interval 6 = the previous seven keyframes).

Training anchors need the full 3 s future; each training query also supervises one extra horizon
drawn from those available in 4..8 s. Validation keeps every keyframe (6019 anchors) and scores each
horizon that is available.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..paths import data_path
from .nuscenes import ByteLRU, NuScenesTables, chronological_groups, select_history

# Prefix of the per-query sampling hash. It is the one used to train the released models, so the same seed
# draws the same histories and extra horizons. Image augmentations are drawn without a seed. The released
# models used sampling_seed = 41 (the default below, equal to train.seed).
SAMPLING_SALT = 'occ3d-native8-v1'


class Occ3DDataset:
    """Occ3D-nuScenes with camera input: streaming frames, windowed queries and their supervision."""
    supervision_mask = 'Occ3D target mask_camera'
    occgt = False

    def __init__(self, metadata_root, image_root, occ_root, split_file, split='train', image_hw=(256, 704),
                 sampling_seed=41, long_horizons=(4, 5, 6, 7, 8), eval_horizons=(1, 2, 3), history_groups=8):
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        if tuple(image_hw) != (256, 704):
            raise ValueError('Expected the 256x704 SparseWorld-TC crop')
        self.split = split
        self.epoch = 0
        self.sampling_seed = int(sampling_seed)
        self.image_hw = tuple(image_hw)
        self.long_horizons = tuple(long_horizons) if split == 'train' else ()
        self.eval_horizons = tuple(eval_horizons)
        self.history_groups = int(history_groups)
        self.tables = NuScenesTables(metadata_root)
        self.image_root = Path(image_root)
        self.occ_root = Path(occ_root)
        names = set(json.loads(Path(split_file).read_text())[split])
        self.raw_scenes = {}
        for scene in self.tables.scene.values():
            if scene['name'] not in names:
                continue
            tokens = []
            token = scene['first_sample_token']
            while token:
                tokens.append(token)
                token = self.tables.sample[token]['next']
            self.raw_scenes[scene['name']] = tokens
        # Anchors: training needs the full 3 s future; validation keeps every keyframe.
        self.scenes = {s: (t[:-6] if split == 'train' else list(t)) for s, t in self.raw_scenes.items()}
        self.scene_names = sorted(self.scenes)
        expected = (700, 23930) if split == 'train' else (150, 6019)
        if (len(self.scenes), sum(map(len, self.scenes.values()))) != expected:
            raise ValueError(f'Occ3D anchor inventory differs: {len(self.scenes)} scenes, '
                             f'{sum(map(len, self.scenes.values()))} anchors (expected {expected})')
        self.capture_cache = ByteLRU(256*1024**2)
        self.augment = False   # training-only image augmentation (augment.py), set by the trainer

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Invalid epoch')
        self.epoch = epoch

    def query_seed(self, token, salt=''):
        """32-bit seed of the random draws of one query (per sampling seed, epoch and keyframe)."""
        raw = f'{SAMPLING_SALT}:{self.sampling_seed}:{self.epoch}:{token}{salt}'.encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little')

    def horizons_at(self, scene, anchor):
        """Supervised/scored horizons (s): the eval horizons available, plus one random available long horizon."""
        rows = self.raw_scenes[scene]
        ahead = len(rows)-1-anchor
        result = [h for h in self.eval_horizons if 2*h <= ahead]
        extra = [h for h in self.long_horizons if 2*h <= ahead and h not in result]
        if extra:
            rng = np.random.RandomState(self.query_seed(rows[anchor], ':horizon'))
            result.append(extra[rng.randint(len(extra))])
        return result

    # ------------------------------------------------------------------ images

    def _load_group(self, tokens, load_rgb=True, transform='bottom_crop'):
        """Six camera images of one capture group with their calibration, resized and cropped."""
        rgb = []
        image_from_world = []
        intrinsics = []
        camera_to_ego = []
        stamps = []
        world = None
        oh, ow = self.image_hw
        for token in tokens:
            sd, k, world_from_ego, cam, proj = self.tables.camera_record(token)
            world = world_from_ego if world is None else world
            if transform == 'resize_055':
                # Augmentation base: the whole image at the largest training scale (no crop).
                resized = (int(sd['width']*.55), int(sd['height']*.55))
                scale = np.diag([resized[0]/sd['width'], resized[1]/sd['height'], 1.])
            elif transform == 'bottom_crop':
                ratio = max(oh/sd['height'], ow/sd['width'])
                resized = (int(sd['width']*ratio), int(sd['height']*ratio))
                left = int(max(0, resized[0]-ow)/2)
                top = resized[1]-oh
                scale = np.array([[ratio, 0., -left], [0., ratio, -top], [0., 0., 1.]])
            else:
                raise ValueError(transform)
            k = scale@k
            proj[:3, :] = scale@proj[:3, :]
            if load_rgb:
                with Image.open(self.image_root/sd['filename']) as image:
                    image = image.convert('RGB')
                    if transform == 'bottom_crop':
                        # Pillow's default resampling (bicubic).
                        image = image.resize(resized).crop((left, top, left+ow, top+oh))
                    else:
                        image = image.resize(resized)
                    rgb.append(np.asarray(image).copy().transpose(2, 0, 1))
            image_from_world.append(proj)
            intrinsics.append(k)
            camera_to_ego.append(cam)
            stamps.append(sd['timestamp']/1e6)
        return dict(rgb=np.stack(rgb) if load_rgb else None, world_from_ego=world,
                    image_from_world=np.stack(image_from_world), intrinsics=np.stack(intrinsics),
                    camera_to_ego=np.stack(camera_to_ego), time=stamps[0])

    def _capture(self, ids, load_rgb=True, transform='bottom_crop'):
        """Cached ``_load_group``."""
        ids = tuple(ids)
        return self.capture_cache.get((ids, load_rgb, transform), lambda: self._load_group(ids, load_rgb, transform))

    # ------------------------------------------------------------------ windowed queries

    def query_spec(self, scene, anchor):
        """Ids of the eight input groups (chronological), the horizons and the target tokens of a windowed query."""
        if not 0 <= anchor < len(self.scenes[scene]):
            raise IndexError(anchor)
        rows = self.raw_scenes[scene]
        token = rows[anchor]
        current = self.tables.group(token)
        rng = np.random.RandomState(self.query_seed(token))
        past = select_history(self.tables.past_groups(token), training=self.split == 'train', rng=rng,
                              count=self.history_groups-1)
        newest_first = [current]+(past or [current]*(self.history_groups-1))
        groups, order = chronological_groups(newest_first, self.tables.sample_data)
        horizons = self.horizons_at(scene, anchor)
        return dict(scene=scene, token=token, anchor_index=anchor, history_mode='window8', window_frame_ids=groups,
                    horizons=horizons, target_tokens={str(h): rows[anchor+2*h] for h in horizons})

    def get_query(self, scene, anchor):
        """Windowed input: the eight capture groups of the anchor's input window and the future query poses."""
        spec = self.query_spec(scene, anchor)
        ids = spec['window_frame_ids']
        if self.augment:
            from .augment import augment_capture
            captures = [augment_capture(self._capture(group, True, 'resize_055'), self.image_hw) for group in ids]
        else:
            captures = [self._capture(group) for group in ids]
        origin = captures[0]['time']
        tokens = [spec['token']]+list(spec['target_tokens'].values())
        poses = np.stack([self.tables.reference_pose(t) for t in tokens]).astype(np.float32)
        stamps = [self.tables.sample[t]['timestamp']/1e6-origin for t in tokens]

        def stack(key):
            return torch.from_numpy(np.stack([r[key] for r in captures])).float()[None]
        batch = {**spec, 'frame_ids': ids,
                 'rgb': torch.from_numpy(np.stack([r['rgb'] for r in captures]))[None],
                 'times': torch.tensor([[r['time']-origin for r in captures]], dtype=torch.float32),
                 'world_from_ego': stack('world_from_ego'), 'image_from_world': stack('image_from_world'),
                 'current_world_from_ego': torch.from_numpy(poses[:1]),
                 'future_world_from_ego': torch.from_numpy(poses[1:])[None],
                 'future_times': torch.tensor([stamps[1:]], dtype=torch.float32)}
        if len(tokens) > 1 and not torch.all(batch['future_times'][:, :1] > batch['times'][:, -1:]):
            raise ValueError('Future query precedes the last permitted observation')
        return batch

    # ------------------------------------------------------------------ streaming frames

    def frame(self, scene, k, load_rgb=True):
        """Streaming input of keyframe k: its capture group, poses, and the future query poses/times."""
        rows = self.raw_scenes[scene]
        token = rows[k]
        origin = self._capture(self.tables.group(rows[0]), False)['time']
        group = self.tables.group(token)
        if self.augment and load_rgb:
            from .augment import augment_capture
            cap = augment_capture(self._capture(group, True, 'resize_055'), self.image_hw)
        else:
            cap = self._capture(group, load_rgb)
        horizons = self.horizons_at(scene, k)
        tokens = [token]+[rows[k+2*h] for h in horizons]
        poses = np.stack([self.tables.reference_pose(t) for t in tokens]).astype(np.float32)
        stamps = [self.tables.sample[t]['timestamp']/1e6-origin for t in tokens]
        return dict(scene=scene, anchor_index=k, token=token, horizons=horizons, frame_ids=[group],
                    history_mode='stream',
                    rgb=torch.from_numpy(np.ascontiguousarray(cap['rgb']))[None, None] if load_rgb else None,
                    times=torch.tensor([[cap['time']-origin]], dtype=torch.float32),
                    world_from_ego=torch.from_numpy(np.asarray(cap['world_from_ego'])).float()[None, None],
                    image_from_world=torch.from_numpy(np.asarray(cap['image_from_world'])).float()[None, None],
                    current_world_from_ego=torch.from_numpy(poses[:1]),
                    future_world_from_ego=torch.from_numpy(poses[1:])[None],
                    future_times=torch.tensor([stamps[1:]], dtype=torch.float32))

    # ------------------------------------------------------------------ supervision

    def label_path(self, scene, token):
        """Occ3D label file of a keyframe."""
        return self.occ_root/scene/token/'labels.npz'

    def _occupancy(self, scene, token):
        with np.load(self.label_path(scene, token)) as occ:
            return occ['semantics'].astype(np.int64), occ['mask_camera'].astype(bool)

    def get_supervision(self, scene, anchor, include_current=True):
        """Labels and camera masks [1,H,X,Y,Z] of the horizons of ``anchor`` (and of the current frame)."""
        rows = self.raw_scenes[scene]
        pairs = [self._occupancy(scene, rows[anchor+2*h]) for h in self.horizons_at(scene, anchor)]
        result = {}
        if pairs:
            result.update(labels=torch.from_numpy(np.stack([p[0] for p in pairs]))[None],
                          mask_camera=torch.from_numpy(np.stack([p[1] for p in pairs]))[None])
        if include_current:
            labels, mask = self._occupancy(scene, rows[anchor])
            result.update(current_labels=torch.from_numpy(labels)[None],
                          current_mask_camera=torch.from_numpy(mask)[None])
        return result


class OccupancyInputDataset(Occ3DDataset):
    """Occ3D anchors with an occupancy grid per keyframe as the only observation.

    ``history_source='gt'``: ground-truth occupancy (PersistWorld-GT); training supervises the full
    volume (``train_mask='full'``). ``history_source='stcocc'``: STCOcc's predictions read from
    ``history_root/<scene>/<token>/labels.npz`` (PersistWorld-STC); targets and masks stay ground truth.
    """
    def __init__(self, *args, history_keyframes=4, train_mask='full', history_source='gt', history_root=None, **kwargs):
        kwargs.pop('history_groups', None)
        super().__init__(*args, **kwargs)
        if train_mask not in ('full', 'camera'):
            raise ValueError('train_mask must be full or camera')
        if history_source != 'gt' and not history_root:
            raise ValueError('A predicted history needs history_root')
        self.history_keyframes = int(history_keyframes)
        self.train_mask = train_mask
        self.history_source = history_source
        self.history_root = Path(history_root) if history_root else None
        self.occgt = history_source == 'gt'
        self.label_cache = ByteLRU(768*1024**2)
        self.pred_cache = ByteLRU(768*1024**2) if history_source != 'gt' else None
        self.supervision_mask = ('full volume' if self.split == 'train' and train_mask == 'full'
                                 else 'Occ3D target mask_camera')

    def _load(self, scene, token):
        with np.load(self.label_path(scene, token)) as occ:
            return occ['semantics'].astype(np.uint8), occ['mask_camera'].astype(bool)

    def _load_pred(self, scene, token):
        path = self.history_root/scene/token/'labels.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing {self.history_source} prediction {path}')
        with np.load(path) as occ:
            sem = occ['semantics']
        if sem.shape != (200, 200, 16):
            raise ValueError(f'Prediction grid {sem.shape} at {path}')
        return sem.astype(np.uint8)

    def _history_grid(self, scene, token):
        if self.history_source == 'gt':
            return self.label_cache.get((scene, token), lambda: self._load(scene, token))[0]
        return self.pred_cache.get((scene, token), lambda: self._load_pred(scene, token))

    def _occupancy(self, scene, token):
        sem, mask = self.label_cache.get((scene, token), lambda: self._load(scene, token))
        return sem.astype(np.int64), mask

    def frame(self, scene, k, load_rgb=True):
        """Keyframe k's occupancy grid [1,1,1,200,200,16] in place of the images, with the label poses."""
        rows = self.raw_scenes[scene]
        token = rows[k]
        stamp = lambda t: self.tables.sample[t]['timestamp']/1e6
        origin = stamp(rows[0])
        horizons = self.horizons_at(scene, k)
        tokens = [token]+[rows[k+2*h] for h in horizons]
        poses = np.stack([self.tables.reference_pose(t) for t in tokens])
        return dict(scene=scene, anchor_index=k, token=token, horizons=horizons, frame_ids=[(token,)],
                    history_mode='stream',
                    rgb=torch.from_numpy(self._history_grid(scene, token))[None, None, None],
                    times=torch.tensor([[stamp(token)-origin]], dtype=torch.float32),
                    world_from_ego=torch.from_numpy(poses[:1].astype(np.float32))[None],
                    image_from_world=torch.from_numpy(np.linalg.inv(poses[:1]).astype(np.float32))[None, :, None],
                    current_world_from_ego=torch.from_numpy(poses[:1].astype(np.float32)),
                    future_world_from_ego=torch.from_numpy(poses[1:].astype(np.float32))[None],
                    future_times=torch.tensor([[stamp(t)-origin for t in tokens[1:]]], dtype=torch.float32))

    def get_supervision(self, scene, anchor, include_current=True):
        result = super().get_supervision(scene, anchor, include_current)
        if self.split == 'train' and self.train_mask == 'full':
            for k in ('mask_camera', 'current_mask_camera'):
                if k in result:
                    result[k] = torch.ones_like(result[k])
        return result


def build_dataset(cfg, split):
    """Dataset of a resolved config (config.load_config)."""
    d = dict(cfg['data'])
    for key in ('metadata_root', 'image_root', 'occ_root', 'split_file', 'history_root'):
        if d.get(key):
            d[key] = data_path(d[key])
    if cfg['input'] == 'camera':
        return Occ3DDataset(**d, split=split)
    return OccupancyInputDataset(**d, split=split, history_source='gt' if cfg['input'] == 'gt' else 'stcocc')
