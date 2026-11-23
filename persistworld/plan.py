"""Update plan of the streaming training (pure Python).

A segment is a run of consecutive keyframes of one scene starting from the initial state, at most
``len(weights)`` (8) writes deep: exactly the life of a state between two resets at inference.
Every update holds 64 loss points from at least 16 segments, rounded up to a multiple of the GPU
count of the plan (18 segments at 6 GPUs). A segment's loss depths (state depths at its supervised
keyframes) are drawn without replacement from 1..8 with ``weights`` (uniform in the paper); the
segment runs up to its deepest loss. The scene is drawn proportionally to its anchor count and the
start uniformly, so every loss frame is a training anchor. An epoch has ceil(anchors / 64) updates
(374 on Occ3D). Draws are seeded per (epoch, update).
"""
import math
import random


def segments_per_update(world, min_segments=16):
    """At least ``min_segments`` segments, rounded up to a multiple of the GPU count."""
    return math.ceil(min_segments/world)*world


def loss_counts(segments, batch=64):
    """Loss points per segment: ``batch`` split as evenly as possible, the first segments one more."""
    return [batch//segments+(i < batch % segments) for i in range(segments)]


def draw_depths(rng, count, weights):
    """``count`` distinct depths (1-based) drawn without replacement with ``weights``."""
    pool = list(range(1, len(weights)+1))
    w = list(weights)
    out = []
    for _ in range(count):
        i = rng.choices(range(len(pool)), weights=w)[0]
        out.append(pool.pop(i))
        w.pop(i)
    return tuple(sorted(out))


# Seed of the update plans of the released models (independent of train.seed, which seeds the initialization).
PLAN_SEED = 1000


def update(scenes, epoch, step, world, batch, min_segments, weights, seed=PLAN_SEED):
    """One update -> per-rank lists of segments (scene, start, loss_depths); segment i on rank i % world."""
    rng = random.Random(f'{seed}:{epoch}:{step}')
    names = sorted(scenes)
    sizes = [scenes[n] for n in names]
    segs = []
    for count in loss_counts(segments_per_update(world, min_segments), batch):
        depths = draw_depths(rng, count, weights)
        deep = depths[-1]
        while True:
            scene = rng.choices(names, weights=sizes)[0]
            if scenes[scene] >= deep:
                break
        segs.append((scene, rng.randrange(scenes[scene]-deep+1), depths))
    return [segs[r::world] for r in range(world)]


def updates_per_epoch(scenes, batch=64):
    """ceil(training anchors / loss points per update)."""
    return math.ceil(sum(scenes.values())/batch)


def segment_cost(seg, frame=0.14, loss=0.64):
    """Measured seconds of one segment on an A800 (0.14 s per frame, 0.64 s per loss)."""
    return frame*max(seg[2])+loss*len(seg[2])


def balance(ranks):
    """The same segments reassigned to ranks by longest processing time first (equal counts per rank)."""
    world = len(ranks)
    per = [len(r) for r in ranks]
    segs = sorted((s for r in ranks for s in r), key=lambda s: (-segment_cost(s), s[0], s[1], s[2]))
    load = [0.]*world
    out = [[] for _ in range(world)]
    for s in segs:
        i = min((i for i in range(world) if len(out[i]) < per[i]), key=lambda i: (load[i], i))
        out[i].append(s)
        load[i] += segment_cost(s)
    return out


def redistribute(ranks, world):
    """The segments of an update planned for len(ranks) GPUs, run on ``world`` GPUs (same update)."""
    segs = sorted((s for r in ranks for s in r), key=lambda s: (-segment_cost(s), s[0], s[1], s[2]))
    load = [0.]*world
    out = [[] for _ in range(world)]
    for s in segs:
        i = min(range(world), key=lambda i: (load[i], i))
        out[i].append(s)
        load[i] += segment_cost(s)
    return out
