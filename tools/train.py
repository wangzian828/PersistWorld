"""Streaming training of PersistWorld (main model, ablations, -STC and -GT variants).

    torchrun --nproc_per_node=6 tools/train.py --variant main --out runs/main

Every update holds 64 loss points from 18 segments, planned as for 6 GPUs (``plan_world``) whatever
the number of GPUs, so fewer GPUs train the same updates. Restarting on an existing --out resumes
from its last.pt. Checkpoints: OUT/epoch_XX.pt (model only) and OUT/last.pt (with the optimizer).
"""
import argparse
import gc
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from persistworld import plan, runtime, streaming as st
from persistworld.build import build_model, trainable_state
from persistworld.config import load_config
from persistworld.loaders.occ3d import build_dataset
from persistworld.training import all_reduce_gradients, learning_rate, make_optimizer, segment_backward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='main', help='model variant (see persistworld/config.py)')
    parser.add_argument('--config', default=None, help='base config (default configs/persistworld_occ3d.json)')
    parser.add_argument('--out', required=True, help='run directory; an existing one resumes from its last.pt')
    parser.add_argument('--epochs', type=int, default=None, help='override train.epochs of the config')
    parser.add_argument('--load-threads', type=int, default=8, help='data loading threads per process')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device = runtime.init_distributed()
    lead = rank == 0
    cfg = load_config(args.variant, args.config)
    t = cfg['train']
    epochs = args.epochs or t['epochs']
    runtime.seed_all(t['seed'])
    model = build_model(cfg).to(device).train()
    if lead:
        runtime.save_json(out/'config.json', cfg)
    optimizer, params = make_optimizer(model, t['lr'], t['backbone_lr_scale'])
    train_ds = build_dataset(cfg, 'train')
    train_ds.augment = bool(t['augment']) and cfg['input'] == 'camera'
    counts = {s: len(x) for s, x in train_ds.scenes.items()}
    batch = t['loss_points_per_update']
    per_epoch = plan.updates_per_epoch(counts, batch)
    weights = tuple(t['depth_weights'])
    window = t['window']
    k = t['tbptt']
    if len(weights) != window:
        raise ValueError('One depth weight per state depth 1..window')
    plan_world = t['plan_world'] or world
    start_epoch, start_step = runtime.resume(out/'last.pt', model, optimizer) if (out/'last.pt').exists() else (0, 0)
    if lead:
        runtime.save_json(out/'protocol.json', dict(
            variant=cfg['variant'], epochs=epochs, world=world, plan_world=plan_world, updates_per_epoch=per_epoch,
            segments_per_update=plan.segments_per_update(plan_world, t['min_segments']),
            trainable_parameters=sum(p.numel() for p in params), resumed_from=[start_epoch, start_step]))
    log = (out/'train.jsonl').open('a') if lead else None
    pool = ThreadPoolExecutor(args.load_threads)

    def load(q):
        scene, j, sup = q
        return (st.pin(train_ds.frame(scene, j)),
                st.pin(train_ds.get_supervision(scene, j, include_current=True)) if sup else None)
    retention = st.Retention(st.KEEP_POSES, window)

    def rank_pass(mine, loaded, sums):
        for (scene, start, depths), items in zip(mine, loaded):
            segment_backward(model, [f for f, _ in items], {d: items[d-1][1] for d in depths}, depths, k, batch,
                             sums, device, retention)

    retries = 0
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        first = start_step if epoch == start_epoch else 0
        if plan_world != world:
            # A different GPU count than planned: the same segments, redistributed.
            work = [(s, plan.redistribute(plan.update(counts, epoch, s, plan_world, batch, t['min_segments'], weights),
                                          world)[rank])
                    for s in range(first, per_epoch)]
        else:
            work = [(s, plan.balance(plan.update(counts, epoch, s, plan_world, batch, t['min_segments'],
                                                 weights))[rank])
                    for s in range(first, per_epoch)]
        # Every frame this rank loads in the epoch: (scene, keyframe, whether it is supervised).
        flat = [(sc, start+j, j+1 in depths)
                for _, mine in work for sc, start, depths in mine for j in range(max(depths))]
        futures = [pool.submit(load, q) for q in flat[:32]]
        cursor = 32
        for step, mine in work:
            global_step = epoch*per_epoch+step
            lr = learning_rate(t['lr'], t['warmup_updates'], global_step, per_epoch, epochs, t['decay_from_epoch'],
                               t['decay_floor'])
            for g in optimizer.param_groups:
                g['lr'] = lr*g['scale']
            optimizer.zero_grad(set_to_none=True)
            start_time = time.monotonic()
            loaded = []
            for _, _, depths in mine:
                seg = []
                for _ in range(max(depths)):
                    seg.append(futures.pop(0).result())
                    if cursor < len(flat):
                        futures.append(pool.submit(load, flat[cursor]))
                        cursor += 1
                loaded.append(seg)
            sums = torch.zeros(4, device=device)
            try:
                rank_pass(mine, loaded, sums)
            except torch.cuda.OutOfMemoryError:
                # Rare memory peak: redo this rank's segments with activation checkpointing (same arithmetic).
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                retries += 1
                saved = (model.evolver.checkpoint_blocks, model.checkpoint_readout, model.writer.checkpoint_chunks)
                model.evolver.checkpoint_blocks = model.checkpoint_readout = model.writer.checkpoint_chunks = True
                try:
                    sums = torch.zeros(4, device=device)
                    rank_pass(mine, loaded, sums)
                finally:
                    model.evolver.checkpoint_blocks, model.checkpoint_readout, model.writer.checkpoint_chunks = saved
            del loaded
            if world > 1:
                all_reduce_gradients(params, sums)
            norm = float(torch.nn.utils.clip_grad_norm_(params, t['grad_clip']))
            if not math.isfinite(norm) or norm >= 1e4:
                raise FloatingPointError(f'Gradient norm {norm}')
            optimizer.step()
            if lead:
                n = float(sums[2].clamp_min(1))
                log.write(json.dumps(dict(epoch=epoch, step=step, global_step=global_step, lr=lr, losses=int(sums[2]),
                                          loss=float(sums[0])/n, ray=float(sums[1])/n, grad_norm=norm,
                                          writes=int(sums[3]), seconds=time.monotonic()-start_time,
                                          oom_retries=retries))+'\n')
                log.flush()
            if (step+1) % 50 == 0 or step+1 == per_epoch:
                if lead:
                    done = step+1 >= per_epoch
                    position = (epoch+1, 0) if done else (epoch, step+1)
                    runtime.save_last(out/'last.pt', model, optimizer, *position)
                if world > 1:
                    torch.distributed.barrier()
        for f in futures:
            f.cancel()
        if lead:
            torch.save(dict(model=trainable_state(model), epoch=epoch+1, config=cfg), out/f'epoch_{epoch+1:02d}.pt')
    if lead:
        runtime.save_json(out/'status.json', dict(stage='complete', epochs=epochs))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
