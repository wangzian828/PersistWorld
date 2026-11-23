"""Latency / FPS of PersistWorld on one GPU (FPS column of Tab. 1 and the speed figure).

    python tools/benchmark_fps.py --ckpt runs/main/epoch_20.pt --out results/fps.json [--path reference|exact|fastest]

Protocol: single-state streaming (reset every 8 groups), batch size one, 4 val scenes; per scene 8
untimed keyframes, then 16 timed keyframes. Timed: from the inputs on the GPU to the uint8 grids of the
current frame and of 1, 2 and 3 s on the CPU (no I/O, no host-to-device copies). One discarded pass
first (compilation, CUDA graphs), then ``--rounds`` measured passes. FPS = 1000 / mean ms per keyframe.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from persistworld import streaming as st
from persistworld.build import build_model, load_weights
from persistworld.config import load_config
from persistworld.loaders.occ3d import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='checkpoint of the main model')
    parser.add_argument('--out', required=True, help='result JSON file')
    parser.add_argument('--path', choices=['reference', 'exact', 'fastest'], default='fastest',
                        help='reference: the evaluation code path; exact: the bitwise-exact switches; '
                             'fastest: the configuration of the paper FPS')
    parser.add_argument('--scenes', type=int, default=4, help='number of val scenes')
    parser.add_argument('--timed', type=int, default=16, help='timed keyframes per scene, after 8 untimed ones')
    parser.add_argument('--rounds', type=int, default=3, help='measured passes, after one discarded pass')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_config('main')
    ds = build_dataset(cfg, 'val')
    ds.eval_horizons = (1, 2, 3)
    model = build_model(cfg)
    load_weights(model, args.ckpt)
    model = model.cuda().eval().requires_grad_(False)
    encode = None
    if args.path != 'reference':
        from persistworld.fast import fastpath
        fastpath.install(model, fastest=args.path == 'fastest')
        encode = lambda m, rgb: fastpath.encode_group(m, rgb)
    names = ds.scene_names[::max(1, len(ds.scene_names)//args.scenes)][:args.scenes]
    frames = {s: [st.pin(ds.frame(s, k)) for k in range(min(8+args.timed, len(ds.scenes[s])-6))] for s in names}

    def labels(x):
        return x[0] if x.dtype == torch.uint8 else x[0].argmax(-1).to(torch.uint8)
    rows = []
    for repeat in range(1+args.rounds):
        runner = st.SingleState(st.PERIOD, encode=encode)
        for s in names:
            for k, cpu in enumerate(frames[s]):
                b = st.transfer(cpu, torch.device('cuda'))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    cur, fut = runner.step_frame(model, b)
                    grids = [labels(x['occupancy']) for x in [cur]+fut]
                out = [g.cpu() for g in grids]
                ms = (time.perf_counter()-t0)*1e3
                if repeat and k >= 8:
                    rows.append(ms)
                del b, out, cur, fut
    x = np.asarray(rows)
    result = dict(path=args.path, frames=len(x), mean_ms=float(x.mean()), p50_ms=float(np.median(x)),
                  fps=float(1e3/x.mean()), gpu=torch.cuda.get_device_name(), scenes=names)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
