"""Windowed training of the same network (Tab. 3 "Windowed"): the state is rebuilt for every query
from the last eight capture groups, written as two blocks of four.

    torchrun --nproc_per_node=6 tools/train_windowed.py --out runs/windowed

Global batch 64 queries, every training anchor once per epoch in a seeded shuffle (374 updates),
the same optimizer, warmup and clipping as the streaming model, cosine decay over epochs 17-20.
"""
import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from persistworld import runtime, streaming as st
from persistworld.build import build_model, trainable_state
from persistworld.config import load_config
from persistworld.loaders.occ3d import build_dataset
from persistworld.training import all_reduce_gradients, learning_rate, make_optimizer, window_loss

QUERIES_PER_UPDATE = 64   # global batch (the streaming model holds 64 loss points per update)
SHUFFLE_SEED = 1000       # per-epoch anchor order of the released windowed model: Random(SHUFFLE_SEED + epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='run directory; an existing one resumes from its last.pt')
    parser.add_argument('--config', default=None, help='base config (default configs/persistworld_occ3d.json)')
    parser.add_argument('--epochs', type=int, default=None, help='override train.epochs of the config')
    # The released windowed model used a floor of 0.3333333 (the streaming models train.decay_floor = 1/3).
    parser.add_argument('--decay-floor', type=float, default=0.3333333,
                        help='final learning rate of the cosine decay, as a fraction of the peak')
    parser.add_argument('--load-threads', type=int, default=8, help='data loading threads per process')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device = runtime.init_distributed()
    lead = rank == 0
    cfg = load_config('main', args.config)
    t = cfg['train']
    epochs = args.epochs or t['epochs']
    if lead:
        runtime.save_json(out/'config.json', cfg)
    runtime.seed_all(t['seed'])
    model = build_model(cfg).to(device).train()
    optimizer, params = make_optimizer(model, t['lr'], t['backbone_lr_scale'])
    train_ds = build_dataset(cfg, 'train')
    train_ds.augment = bool(t['augment'])
    anchors = [(s, x) for s in train_ds.scene_names for x in range(len(train_ds.scenes[s]))]
    per_epoch = math.ceil(len(anchors)/QUERIES_PER_UPDATE)
    start_epoch, start_step = runtime.resume(out/'last.pt', model, optimizer) if (out/'last.pt').exists() else (0, 0)
    log = (out/'train.jsonl').open('a') if lead else None
    pool = ThreadPoolExecutor(args.load_threads)

    def load(q):
        return st.pin(train_ds.get_query(*q)), st.pin(train_ds.get_supervision(*q, include_current=True))
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)   # per-epoch history interval and long-horizon draws
        plan = random.Random(SHUFFLE_SEED+epoch).sample(anchors, len(anchors))
        first = start_step if epoch == start_epoch else 0
        n = QUERIES_PER_UPDATE
        mine = [q for s in range(first, per_epoch) for q in plan[s*n:(s+1)*n][rank::world]]
        futures = [pool.submit(load, q) for q in mine[:16]]
        cursor = 16
        for step in range(first, per_epoch):
            global_step = epoch*per_epoch+step
            count = len(plan[step*n:(step+1)*n])
            lr = learning_rate(t['lr'], t['warmup_updates'], global_step, per_epoch, epochs, t['decay_from_epoch'],
                               args.decay_floor)
            for g in optimizer.param_groups:
                g['lr'] = lr*g['scale']
            optimizer.zero_grad(set_to_none=True)
            start_time = time.monotonic()
            sums = torch.zeros(3, device=device)
            for _ in plan[step*n:(step+1)*n][rank::world]:
                batch, sup = futures.pop(0).result()
                if cursor < len(mine):
                    futures.append(pool.submit(load, mine[cursor]))
                    cursor += 1
                loss, ray = window_loss(model, batch, sup, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss/count).backward()
                ray = loss.detach()*0 if ray is None else ray
                sums += torch.stack([loss.detach().float(), ray.float(), torch.ones((), device=device)])
            if world > 1:
                all_reduce_gradients(params, sums)
            norm = float(torch.nn.utils.clip_grad_norm_(params, t['grad_clip']))
            if not math.isfinite(norm) or norm >= 1e4:
                raise FloatingPointError(f'Gradient norm {norm}')
            optimizer.step()
            if lead:
                log.write(json.dumps(dict(epoch=epoch, step=step, lr=lr, queries=int(sums[2]),
                                          loss=float(sums[0]/sums[2]), ray=float(sums[1]/sums[2]), grad_norm=norm,
                                          seconds=time.monotonic()-start_time))+'\n')
                log.flush()
            if (step+1) % 50 == 0 or step+1 == per_epoch:
                if lead:
                    done = step+1 == per_epoch
                    position = (epoch+1, 0) if done else (epoch, step+1)
                    runtime.save_last(out/'last.pt', model, optimizer, *position)
                if world > 1:
                    torch.distributed.barrier()
        if lead:
            torch.save(dict(model=trainable_state(model), epoch=epoch+1, config=cfg, mode='windowed'),
                       out/f'epoch_{epoch+1:02d}.pt')
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
