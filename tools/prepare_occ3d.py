"""Prepare and check the Occ3D-nuScenes layout under the data root ($PERSISTWORLD_DATA, default ./data).

    python tools/prepare_occ3d.py            # writes occ3d_nuscenes/official_splits.json, checks the layout

Expected layout (see README):
    occ3d_nuscenes/raw_nuscenes/extracted/v1.0-trainval/*.json    nuScenes metadata
    occ3d_nuscenes/raw_nuscenes/extracted/samples/CAM_*/          keyframe images
    occ3d_nuscenes/raw_nuscenes/extracted/sweeps/CAM_*/           sweep images (camera history)
    occ3d_nuscenes/extracted/gts/<scene>/<token>/labels.npz       Occ3D labels (semantics, mask_camera)
    models/dinov3-vitb16-pretrain-lvd1689m/{config.json,model.safetensors}
    stcocc/pred_occ3d/<scene>/<token>/labels.npz                  (PersistWorld-STC only) STCOcc predictions
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persistworld.loaders.nuscenes import CAMERAS
from persistworld.paths import DATA_ROOT


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = DATA_ROOT/'occ3d_nuscenes'
    split = root/'official_splits.json'
    if not split.exists():
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError:
            sys.exit('pip install nuscenes-devkit (only needed once, to write the official scene split)')
        scenes = create_splits_scenes()
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text(json.dumps(dict(train=sorted(scenes['train']), val=sorted(scenes['val']))))
    names = json.loads(split.read_text())
    print('split', {k: len(v) for k, v in names.items()})
    raw = root/'raw_nuscenes/extracted'
    checks = [raw/'v1.0-trainval'/f'{t}.json'
              for t in ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'scene')]
    checks += [raw/'samples'/c for c in CAMERAS]
    checks += [DATA_ROOT/'models/dinov3-vitb16-pretrain-lvd1689m'/f for f in ('config.json', 'model.safetensors')]
    missing = [str(p) for p in checks if not p.exists()]
    if not all((raw/'sweeps'/c).exists() for c in CAMERAS):
        print('note: no sweep images; they are only needed for the windowed model (tools/train_windowed.py)')
    gts = root/'extracted/gts'
    scenes = sum(1 for _ in gts.iterdir()) if gts.is_dir() else 0
    print('Occ3D label scenes:', scenes, '(850 expected)')
    if missing or scenes != 850:
        sys.exit('missing:\n  '+'\n  '.join(missing) if missing else 'incomplete Occ3D labels')
    print('layout OK')


if __name__ == '__main__':
    main()
