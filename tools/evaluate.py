"""Evaluation on the Occ3D-nuScenes val split.

Streaming (Tab. 1, 2, 3 and the appendix): one state per scene, reset after every ``--period``
capture groups (8) and at scene starts; every keyframe is answered with the state it holds.

    python tools/evaluate.py eval --ckpt runs/main/epoch_20.pt --out results/main.json
    python tools/evaluate.py eval --variant gt --ckpt runs/gt/epoch_20.pt --out results/gt.json    # Occ-GT protocol
    python tools/evaluate.py eval --variant gt --ckpt runs/gt/epoch_20.pt --protocol come8 --out results/gt_come8.json
    python tools/evaluate.py eval --variant gt --baseline copy_forward --out results/gt_warp.json    # GT + warp rows
    python tools/evaluate.py windowed --ckpt runs/windowed/epoch_20.pt --out results/windowed.json    # Tab. 3

The horizons default to 1..8 s (``--horizons``).

Large runs can be split with ``--shard i/N`` (by scene, one process per GPU) and merged with
``python tools/evaluate.py merge --out results/main.json results/main.shard*.json``.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from persistworld import eval_protocol as ep
from persistworld import streaming as st
from persistworld.build import build_model, load_weights
from persistworld.config import load_config
from persistworld.loaders.occ3d import build_dataset
from persistworld.metrics import confusion_matrix, details, summary_lines
from persistworld.training import window_inputs

# Occ3D grid of the copy-forward baseline: shape, lower corner (m), voxel size (m) and free class.
SHAPE = (200, 200, 16)
ORIGIN = np.array([-40., -40., -1.])
VOXEL = .4
FREE = 17


def copy_forward(ds, scene, k, horizons, sup):
    """Baseline '+ warp': the current input grid moved into each target frame with the ego poses
    (nearest voxel centre, free outside the input grid); horizon 0 is the input grid itself."""
    rows = ds.raw_scenes[scene]
    pose = ds.tables.reference_pose
    grid = (sup['current_labels'][0].numpy() if ds.history_source == 'gt'
            else ds._history_grid(scene, rows[k])).astype(np.uint8)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing='ij'), -1).reshape(-1, 3)
    centres = np.concatenate([ORIGIN+(idx+.5)*VOXEL, np.ones((len(idx), 1))], 1)
    out = [grid]
    for h in horizons:
        p = centres@np.linalg.solve(pose(rows[k]), pose(rows[k+2*h])).T
        ijk = np.floor((p[:, :3]-ORIGIN)/VOXEL).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.array(SHAPE)), 1)
        warped = np.full(len(p), FREE, np.uint8)
        warped[inside] = grid.reshape(-1)[np.ravel_multi_index(ijk[inside].T, SHAPE)]
        out.append(warped.reshape(SHAPE))
    return out


def setup(args):
    """Config, model (None for a baseline) and val dataset of the command line."""
    cfg = load_config(args.variant)
    ds = build_dataset(cfg, 'val')
    ds.eval_horizons = tuple(int(x) for x in args.horizons.split(','))
    if args.baseline:
        return cfg, None, ds
    torch.set_num_threads(4)
    model = build_model(cfg)
    load_weights(model, args.ckpt)
    model = model.cuda().eval().requires_grad_(False)
    return cfg, model, ds


def evaluate(args):
    """Streaming evaluation: every keyframe in order, scored under the selected protocol."""
    cfg, model, ds = setup(args)
    protocol = ep.resolve(args.protocol, ds)
    names = ds.scene_names[:args.scenes] if args.scenes else ds.scene_names
    shard, shards = map(int, args.shard.split('/'))
    names = names[shard::shards]
    jobs = [(s, k) for s in names for k in range(ep.stream_length(protocol, ds, s))]
    if model is None:   # a baseline keeps no state: only the scored anchors are loaded
        jobs = [(s, k) for s, k in jobs if ep.scored(protocol, ds, s, k)]
    nh = max(ds.eval_horizons)+1
    nc = cfg['model']['num_classes']
    free = cfg['model']['free_id']
    cm = np.zeros((nh, nc, nc), np.int64)
    by_depth = {}
    runner = st.SingleState(args.period) if model is not None else None
    dumped = {}
    scored = 0

    def load(q):
        cpu = ds.frame(*q, load_rgb=model is not None)
        return (st.pin(cpu) if model is not None else cpu), ds.get_supervision(*q, include_current=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(4) as pool, torch.no_grad():
        pending = [pool.submit(load, q) for q in jobs[:8]]
        cursor = len(pending)
        for done, (scene, k) in enumerate(jobs, 1):
            cpu, sup = pending.pop(0).result()
            if cursor < len(jobs):
                pending.append(pool.submit(load, jobs[cursor]))
                cursor += 1
            score = ep.scored(protocol, ds, scene, k)
            horizons = [0]+list(cpu['horizons'])
            if model is not None:
                batch = st.transfer(cpu, torch.device('cuda'))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    cur, fut = runner.step_frame(model, batch)
                logits = [cur['occupancy']]+[x['occupancy'] for x in fut]
                depth = runner.stream.session.updates
                preds = [lg[0].argmax(-1).cpu().numpy() for lg in logits] if score or args.dump else None
            else:
                depth = None
                preds = copy_forward(ds, scene, k, cpu['horizons'], sup)
            ys = [sup['current_labels'], *(sup['labels'].unbind(1) if 'labels' in sup else [])]
            if score:
                scored += 1
                dcm = by_depth.setdefault(depth, np.zeros_like(cm))
                for h, pred, y, mask in zip(horizons, preds, ys, ep.masks(protocol, sup)):
                    one = confusion_matrix(pred, y[0].numpy(), mask[0].numpy(), nc)
                    cm[h] += one
                    dcm[h] += one
            if args.dump and preds is not None:
                for h, pred in zip(horizons, preds):
                    if h <= 3:
                        dumped.setdefault(scene, {})[f"{cpu['token']}_{h}"] = pred.astype(np.uint8)
                if k == ep.stream_length(protocol, ds, scene)-1 and scene in dumped:
                    Path(args.dump).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(Path(args.dump)/f'{scene}.npz', **dumped.pop(scene))
            if done % 200 == 0 or done == len(jobs):
                print(json.dumps(dict(done=done, total=len(jobs), seconds=round(time.monotonic()-t0))), flush=True)
    if scored != ep.count(protocol, ds, names):
        raise AssertionError('Scored anchors differ from the protocol')
    result = dict(variant=cfg['variant'], checkpoint=args.ckpt, baseline=args.baseline or None, protocol=protocol,
                  period=args.period, anchors=scored, scenes=names, shard=[shard, shards],
                  horizons=list(ds.eval_horizons), metrics=details(cm, free), confusion=cm.tolist(),
                  by_depth={str(d): v.tolist() for d, v in by_depth.items()})
    write(args.out, result)


def evaluate_windowed(args):
    """The windowed model: every val keyframe answered from a state rebuilt from its eight-group window."""
    args.baseline = ''
    cfg, model, ds = setup(args)
    names = ds.scene_names[:args.scenes] if args.scenes else ds.scene_names
    shard, shards = map(int, args.shard.split('/'))
    names = names[shard::shards]
    jobs = [(s, k) for s in names for k in range(len(ds.scenes[s]))]
    nh = max(ds.eval_horizons)+1
    nc = cfg['model']['num_classes']
    cm = np.zeros((nh, nc, nc), np.int64)

    def load(q):
        return ds.get_query(*q), ds.get_supervision(*q, include_current=True)
    with ThreadPoolExecutor(4) as pool, torch.no_grad():
        pending = [pool.submit(load, q) for q in jobs[:8]]
        cursor = len(pending)
        for _ in jobs:
            batch, sup = pending.pop(0).result()
            if cursor < len(jobs):
                pending.append(pool.submit(load, jobs[cursor]))
                cursor += 1
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fut, cur = model.forward_query(window_inputs(batch, torch.device('cuda')), current_readout=True)
            logits = [cur['occupancy']]+[x['occupancy'] for x in fut]
            ys = [sup['current_labels'], *(sup['labels'].unbind(1) if 'labels' in sup else [])]
            ms = [sup['current_mask_camera'], *(sup['mask_camera'].unbind(1) if 'mask_camera' in sup else [])]
            for h, lg, y, m in zip([0]+list(batch['horizons']), logits, ys, ms):
                if h < nh:
                    cm[h] += confusion_matrix(lg[0].argmax(-1).cpu().numpy(), y[0].numpy(), m[0].numpy(), nc)
    write(args.out, dict(variant='windowed', checkpoint=args.ckpt, protocol='camera', anchors=len(jobs),
                         scenes=names, shard=[shard, shards], horizons=list(ds.eval_horizons),
                         metrics=details(cm, cfg['model']['free_id']), confusion=cm.tolist()))


def write(path, result):
    """Save the result JSON and print the summary (mIoU / IoU per horizon)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result))
    print('\n'.join(summary_lines(result['metrics'])))


def merge(args):
    """Combine the shard results of one evaluation (sums of the confusion matrices)."""
    parts = [json.loads(Path(f).read_text()) for f in args.shards]
    n = {r['shard'][1] for r in parts}
    if len(n) != 1 or sorted(r['shard'][0] for r in parts) != list(range(n.pop())):
        raise ValueError('Missing or duplicate shards')
    for key in ('variant', 'checkpoint', 'protocol', 'horizons'):
        if len({json.dumps(r.get(key)) for r in parts}) != 1:
            raise ValueError(f'Shards differ in {key}')
    cm = sum(np.asarray(r['confusion'], np.int64) for r in parts)
    out = dict(parts[0], shard=None, anchors=sum(r['anchors'] for r in parts),
               scenes=[s for r in parts for s in r['scenes']], metrics=details(cm), confusion=cm.tolist())
    if all('by_depth' in r for r in parts):
        depth = {}
        for r in parts:
            for d, v in r['by_depth'].items():
                depth[d] = depth.get(d, 0)+np.asarray(v, np.int64)
        out['by_depth'] = {d: v.tolist() for d, v in depth.items()}
    write(args.out, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['eval', 'windowed', 'merge'],
                        help='eval: streaming evaluation; windowed: the windowed model; merge: combine shard results')
    parser.add_argument('shards', nargs='*', help='merge: shard result files')
    parser.add_argument('--variant', default='main', help='model variant (see persistworld/config.py)')
    parser.add_argument('--ckpt', default='', help='checkpoint (not needed with --baseline)')
    parser.add_argument('--out', required=True, help='result JSON file')
    parser.add_argument('--horizons', default='1,2,3,4,5,6,7,8', help='comma-separated forecast horizons in seconds')
    parser.add_argument('--period', type=int, default=st.PERIOD,
                        help='reset period of the state (8; 1..16 for the reset-period figure)')
    parser.add_argument('--protocol', choices=['auto', 'camera', 'occgt', 'come8'], default='auto',
                        help='auto: occgt for ground-truth occupancy input, camera otherwise')
    parser.add_argument('--baseline', choices=['', 'copy_forward'], default='',
                        help='score the input occupancy warped with the ego poses instead of a model '
                             '(gt and stc variants)')
    parser.add_argument('--scenes', type=int, default=0, help='first N val scenes only (quick checks)')
    parser.add_argument('--shard', default='0/1', help='i/N: evaluate every N-th scene, starting at scene i')
    parser.add_argument('--dump', default='',
                        help='directory for argmax predictions <scene>.npz (keys <token>_<h>, h = 0..3, uint8)')
    args = parser.parse_intermixed_args()   # allows `merge --out FILE SHARD...`
    if args.mode == 'merge':
        merge(args)
    elif args.mode == 'windowed':
        evaluate_windowed(args)
    else:
        if not args.baseline and not args.ckpt:
            parser.error('--ckpt is required')
        evaluate(args)
