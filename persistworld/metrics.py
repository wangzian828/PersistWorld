"""Occupancy metrics from confusion matrices.

Confusions count (ground truth, prediction) pairs over the evaluated voxels of all target frames of one horizon.
The semantic mIoU averages the IoU of the non-free classes that occur in the ground truth; the geometric IoU scores
occupied against free.
"""
import numpy as np


def confusion_matrix(pred, gt, mask, num_classes):
    """Counts [C, C] (rows: ground truth, columns: prediction) over the voxels selected by ``mask``."""
    pred, gt, mask = np.asarray(pred, np.int64), np.asarray(gt, np.int64), np.asarray(mask, bool)
    if not pred.shape == gt.shape == mask.shape:
        raise ValueError(f'Shapes differ: prediction {pred.shape}, ground truth {gt.shape}, mask {mask.shape}')
    keep = mask & (gt >= 0) & (gt < num_classes)
    p, g = pred[keep], gt[keep]
    if p.size and (p.min() < 0 or p.max() >= num_classes):
        raise ValueError(f'Predicted class ids outside [0, {num_classes})')
    return np.bincount(g*num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes).astype(np.int64)


def iou_scores(confusion, free_id):
    """-> (per-class IoU [C], NaN for classes absent from the ground truth; geometric IoU; semantic mIoU)."""
    cm = np.asarray(confusion, np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f'Confusion matrix must be square, got {cm.shape}')
    tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.where(gt > 0, tp/(gt + pred - tp), np.nan)
    occupied = np.arange(len(cm)) != free_id
    both = cm[np.ix_(occupied, occupied)].sum()
    union = cm[occupied].sum() + cm[:, occupied].sum() - both
    geometric = both/union if union > 0 else np.nan
    classes = per_class[occupied]
    semantic = np.nanmean(classes) if np.isfinite(classes).any() else np.nan
    return per_class, float(geometric), float(semantic)


def details(cm, free=17):
    """Per-horizon metrics (percent) of confusions cm [H,C,C]; horizon 0 is the current frame."""
    out = {}
    for h, m in enumerate(cm):
        per, geo, sem = iou_scores(m, free)
        out[str(h)] = dict(semantic_miou_percent=float(sem*100), geometry_iou_percent=float(geo*100),
                           per_class_iou_percent=[None if np.isnan(x) else float(x*100) for x in per],
                           voxels=int(np.asarray(m).sum()))
    return dict(per_horizon=out)


def summary_lines(metrics):
    """Text lines: current frame, each horizon, and the 1-3 s and 1-8 s averages (mIoU / IoU)."""
    ph = metrics['per_horizon']
    hs = [h for h in ph if int(h) > 0 and ph[h]['voxels']]

    def avg(sel):
        return ' / '.join('%.2f' % (sum(ph[h][k] for h in sel)/len(sel))
                          for k in ('semantic_miou_percent', 'geometry_iou_percent'))
    lines = ['current  ' + avg(['0'])] + [f'{h} s      ' + avg([h]) for h in hs]
    short = [h for h in hs if int(h) <= 3]
    if short:
        lines.append('1-3 s    ' + avg(short))
    if {int(h) for h in hs} >= set(range(1, 9)):
        lines.append('1-8 s    ' + avg([str(h) for h in range(1, 9)]))
    return lines
