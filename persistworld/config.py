"""Model variants of the paper, as modifications of ``configs/persistworld_occ3d.json``.

    main                 PersistWorld (Tab. 1, Tab. 2)
    stc                  written from the occupancy predicted by STCOcc (Tab. 3 "From STCOcc", Appendix B)
    gt                   written from ground-truth occupancy (Tab. 1 Occ-GT block, Appendix B)
    abl_ego_state        Tab. 3 "Ego-centric state": the state lattice moves with the ego
    abl_frozen_encoder   Tab. 3 "Frozen encoder": DINOv3 not fine-tuned
    abl_no_evolver       Tab. 3 "No Evolver": the state is read unchanged at the queried pose
    abl_no_depth         Appendix C "No depth conditioning": ray depth supervised but not used for writing
    abl_mean_writing     Appendix C "Mean writing": the writer averages its samples instead of attending
    abl_no_context       Appendix C "No shared context": the evolver without the shared observed context

The windowed model of Tab. 3 is the main network trained by tools/train_windowed.py.
"""
import json
from pathlib import Path

from .paths import CONFIG_DIR

VARIANTS = ('main', 'stc', 'gt', 'abl_ego_state', 'abl_frozen_encoder', 'abl_no_evolver', 'abl_no_depth',
            'abl_mean_writing', 'abl_no_context')
# Default location of the STCOcc predictions (<scene>/<token>/labels.npz) under the data root.
STCOCC_PREDICTIONS = 'stcocc/pred_occ3d'


def load_config(variant='main', path=None):
    """The resolved config of ``variant`` (data paths stay relative to the data root)."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {VARIANTS}')
    cfg = json.loads(Path(path or CONFIG_DIR/'persistworld_occ3d.json').read_text())
    cfg['variant'] = variant
    if variant in ('stc', 'gt'):
        # Occupancy input: 4 past keyframes + the current one at 2 Hz in windowed form; in streaming, each
        # keyframe's grid is written once. Training mask: the full volume for GT input (as the Occ-GT
        # methods), the camera mask for STCOcc input (as the camera model).
        cfg['input'] = 'gt' if variant == 'gt' else 'stcocc'
        cfg.pop('encoder')
        cfg['data'].pop('history_groups')
        cfg['data'].update(history_keyframes=4, train_mask='full' if variant == 'gt' else 'camera')
        if variant == 'stc':
            cfg['data']['history_root'] = STCOCC_PREDICTIONS
    elif variant == 'abl_frozen_encoder':
        cfg['encoder']['tune_blocks'] = 0
    elif variant == 'abl_mean_writing':
        cfg['ablation'] = 'mean_writing'
    elif variant == 'abl_no_depth':
        cfg['ablation'] = 'no_depth'
    elif variant == 'abl_ego_state':
        cfg['ablation'] = 'ego_state'
    elif variant == 'abl_no_evolver':
        cfg['ablation'] = 'no_evolver'
    elif variant == 'abl_no_context':
        cfg['ablation'] = 'no_context'
    return cfg
