"""Which keyframes the streaming evaluation scores, and over which voxels.

``camera`` (camera and STCOcc input): every Occ3D val keyframe (6019 anchors), inside the target mask_camera.
``occgt`` (Occ-GT block of Tab. 1): keyframes with 2 s of history (4 earlier keyframes) and 3 s of
  future inside the scene (4519 anchors), the full 200x200x16 volume, single frames at 1/2/3 s.
``come8`` (long-horizon Occ-GT setting of COME, Appendix B): one anchor per val scene, its 4th
  keyframe (1.5 s of history), the full volume, single frames at 1..8 s. The stream stops at the
  anchor, so the state holds exactly the four input keyframes.

The state consumes every keyframe in order either way; a protocol only selects what is scored.
"""
PROTOCOLS = ('camera', 'occgt', 'come8')
OCCGT_HISTORY_KEYFRAMES = 4   # 2 s at 2 Hz before the anchor
OCCGT_FUTURE_S = 3
COME_ANCHOR = 3               # frames 0-3 condition, frame 3 is the current frame
COME_FUTURE_S = 8


def resolve(name, ds):
    """'auto': occgt for ground-truth occupancy input, camera otherwise."""
    if name in (None, '', 'auto'):
        name = 'occgt' if getattr(ds, 'occgt', False) else 'camera'
    if name not in PROTOCOLS:
        raise ValueError(f'Unknown evaluation protocol {name!r}; expected one of {PROTOCOLS}')
    return name


def scored(name, ds, scene, k):
    """Is keyframe k of ``scene`` an evaluation anchor of the protocol?"""
    if name == 'camera':
        return True
    if name == 'come8':
        return k == COME_ANCHOR and k+2*COME_FUTURE_S <= len(ds.raw_scenes[scene])-1
    return k >= OCCGT_HISTORY_KEYFRAMES and k+2*OCCGT_FUTURE_S <= len(ds.raw_scenes[scene])-1


def masks(name, sup):
    """Scored voxels per target (current first, then the future horizons)."""
    ms = [sup['current_mask_camera'], *(sup['mask_camera'].unbind(1) if 'mask_camera' in sup else [])]
    if name in ('occgt', 'come8'):
        import torch
        ms = [torch.ones_like(m) for m in ms]
    return ms


def stream_length(name, ds, scene):
    """How many keyframes of ``scene`` the state consumes (come8 stops at its anchor)."""
    n = len(ds.scenes[scene])
    return min(n, COME_ANCHOR+1) if name == 'come8' else n


def count(name, ds, scenes):
    """Number of anchors the protocol scores in ``scenes``."""
    return sum(scored(name, ds, s, k) for s in scenes for k in range(len(ds.scenes[s])))
