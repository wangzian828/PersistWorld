"""Training-only image augmentation with exact projection updates (as SparseWorld-TC / BEVDet).

Each camera image gets its own random scale in [0.38, 0.55] of the original resolution, a
bottom-aligned crop with a random horizontal offset, a random horizontal flip and photometric
distortion (brightness, contrast, saturation, each with p=0.5; no hue). The same pixel affine is
applied to the image's ``image_from_world`` and intrinsics, so every 3D point still lands on the
same image content. Inputs come from the ``resize_055`` transform (whole image at 0.55).
"""
import numpy as np
from PIL import Image

BASE = .55
RESIZE = (.38, .55)


def _photometric(img, rng):
    """img float32 HWC in [0,255]."""
    if rng.random() < .5:
        img = img+rng.uniform(-32, 32)
    contrast_first = rng.random() < .5

    def contrast(x):
        return x*rng.uniform(.5, 1.5) if rng.random() < .5 else x
    if contrast_first:
        img = contrast(img)
    if rng.random() < .5:
        gray = img.mean(-1, keepdims=True)
        img = gray+(img-gray)*rng.uniform(.5, 1.5)
    if not contrast_first:
        img = contrast(img)
    return np.clip(img, 0, 255)


def augment_image(rgb, proj, k, out_hw, rng):
    """rgb [3,H,W] uint8 at BASE scale; proj [4,4] image_from_world; k [3,3] -> augmented copies."""
    oh, ow = out_hw
    _, h, w = rgb.shape
    r = rng.uniform(*RESIZE)
    f = r/BASE
    new_w, new_h = int(round(w*f)), int(round(h*f))
    top = new_h-oh                                   # bottom-aligned
    left = int(rng.uniform(0, max(0, new_w-ow)))
    flip = rng.random() < .5
    image = (Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 2, 0)))
             .resize((new_w, new_h), Image.Resampling.BILINEAR))
    image = image.crop((left, top, left+ow, top+oh))
    a = np.array([[new_w/w, 0., -left], [0., new_h/h, -top], [0., 0., 1.]])
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        a = np.array([[-1., 0., ow-1.], [0., 1., 0.], [0., 0., 1.]])@a
    out = _photometric(np.asarray(image, dtype=np.float32), rng)
    proj = proj.copy()
    proj[:3, :] = a@proj[:3, :]
    return out.round().astype(np.uint8).transpose(2, 0, 1), proj, a@k


def augment_capture(capture, out_hw, rng=None):
    """New capture dict (the cached input is never modified)."""
    rng = np.random.default_rng() if rng is None else rng
    rgb, proj, k = [], [], []
    for i in range(capture['rgb'].shape[0]):
        x, p, kk = augment_image(capture['rgb'][i], capture['image_from_world'][i], capture['intrinsics'][i],
                                 out_hw, rng)
        rgb.append(x)
        proj.append(p)
        k.append(kk)
    return dict(capture, rgb=np.stack(rgb), image_from_world=np.stack(proj), intrinsics=np.stack(k))
