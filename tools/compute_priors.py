"""Class priors of the occupancy objective from the Occ3D training split (CPU only).

Per training query: the masked class frequencies of the current grid (weight 0.25) and the mean of the
1/2/3 s targets, averaged within each scene (``--per-scene`` anchors spread evenly), then over scenes.
The result is ``objective.class_priors`` of configs/persistworld_occ3d.json.

    python tools/compute_priors.py --per-scene 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from persistworld.config import load_config
from persistworld.loaders.occ3d import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-scene', type=int, default=4, help='anchors per training scene, spread evenly')
    args = parser.parse_args()
    cfg = load_config('main')
    cfg['data']['long_horizons'] = []
    ds = build_dataset(cfg, 'train')

    def freq(scene, token):
        labels, mask = ds._occupancy(scene, token)
        return np.bincount(labels[mask], minlength=18)[:18]/max(int(mask.sum()), 1)
    scenes = []
    for s in ds.scene_names:
        n = len(ds.scenes[s])
        rows = ds.raw_scenes[s]
        picks = sorted({int(i) for i in np.linspace(0, n-1, args.per_scene)})
        q = [(.25*freq(s, rows[i])+np.mean([freq(s, rows[i+2*h]) for h in (1, 2, 3)], 0))/1.25 for i in picks]
        scenes.append(np.mean(q, 0))
    priors = np.mean(scenes, 0)
    priors = priors/priors.sum()
    print(json.dumps(dict(class_priors=[float(x) for x in priors], per_scene=args.per_scene, scenes=len(scenes))))


if __name__ == '__main__':
    main()
