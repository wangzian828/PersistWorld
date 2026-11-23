"""DINOv3 ViT-B/16 image encoder with the last blocks fine-tuned.

Each image is encoded independently (no temporal attention); CLS and register tokens are dropped
and the patch tokens of the final normalized layer feed the writer's key and value projections.
The first blocks are frozen and run without gradient; the last ``tune_blocks`` blocks and the
final norm are trained with the world model (FP32 master weights under BF16 autocast).
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

ARCHITECTURES = {
    'dinov3_vitb16': dict(hidden_size=768, num_attention_heads=12, intermediate_size=3072),
}


def file_sha256(path):
    """SHA-256 hex digest of a file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


class DINOv3ImageEncoder(nn.Module):
    """The pretrained DINOv3 checkpoint (verified by digest) with its last ``tune_blocks`` blocks trainable."""
    def __init__(self, checkpoint, *, weights_sha256, config_sha256, image_batch_size=6, dtype=torch.bfloat16,
                 kind='dinov3_vitb16', tune_blocks=0, verify_checksums=True):
        super().__init__()
        if kind not in ARCHITECTURES:
            raise ValueError('Unknown DINOv3 architecture: ' + str(kind))
        self.feature_dim = ARCHITECTURES[kind]['hidden_size']
        from transformers import DINOv3ViTModel
        path = Path(checkpoint)
        for name, expected in [('config.json', config_sha256), ('model.safetensors', weights_sha256)]:
            if not (path / name).is_file():
                raise FileNotFoundError('DINOv3 file required: ' + str(path / name))
            if verify_checksums and file_sha256(path / name) != expected:
                raise ValueError(f'DINOv3 checkpoint digest differs: {name} (expected the official '
                                 'facebook/dinov3-vitb16-pretrain-lvd1689m release)')
        cfg = json.loads((path / 'config.json').read_text())
        expected = dict(model_type='dinov3_vit', num_hidden_layers=12, patch_size=16,
                        num_register_tokens=4, num_channels=3, **ARCHITECTURES[kind])
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError('Checkpoint is not the official DINOv3 architecture for ' + kind)
        if type(image_batch_size) is not int or image_batch_size < 1:
            raise ValueError('Positive image batch size required')
        self.image_batch_size = image_batch_size
        self.model, info = DINOv3ViTModel.from_pretrained(
            str(path), local_files_only=True, dtype=dtype,
            attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('DINOv3 checkpoint did not load exactly: ' + str(info))
        self.model.eval().requires_grad_(False)
        if type(tune_blocks) is not int or not 0 <= tune_blocks <= len(self.model.layer):
            raise ValueError('tune_blocks must be an integer within the block count')
        self.tune_blocks = tune_blocks
        for module in (list(self.model.layer)[len(self.model.layer)-tune_blocks:]+[self.model.norm]
                       if tune_blocks else []):
            module.float().requires_grad_(True)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        # The backbone always runs in eval mode; only the requires_grad flags differ between the blocks.
        super().train(mode)
        self.model.eval()
        return self

    def _tuned_forward(self, pixels):
        """Frozen blocks without gradient, then the tuned blocks and the final norm under BF16 autocast."""
        m = self.model
        split = len(m.layer)-self.tune_blocks
        with torch.no_grad():
            hidden = m.embeddings(pixels.to(m.embeddings.patch_embeddings.weight.dtype))
            position = m.rope_embeddings(pixels)
            for layer in m.layer[:split]:
                hidden = layer(hidden, position_embeddings=position)
        with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for layer in m.layer[split:]:
                hidden = layer(hidden, position_embeddings=position)
            return m.norm(hidden)

    def extract_images(self, rgb):
        """uint8 [I,3,H,W] -> patch tokens [I,P,C] (without gradient when no block is tuned)."""
        if not self.tune_blocks:
            with torch.no_grad():
                return self._extract_images(rgb)
        return self._extract_images(rgb)

    def _extract_images(self, rgb):
        if rgb.ndim != 4 or rgb.dtype != torch.uint8:
            raise ValueError('Expected uint8 [images,3,H,W]')
        count, channels, height, width = rgb.shape
        if count < 1 or channels != 3 or min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError('Nonempty RGB images with patch-aligned dimensions required')
        if self.model.training or (not self.tune_blocks and any(p.requires_grad for p in self.model.parameters())):
            raise ValueError('DINOv3 image features require an eval backbone, frozen unless tune_blocks')
        dtype = next(self.model.parameters()).dtype
        prefix = 1 + self.model.config.num_register_tokens
        outputs = []
        for start in range(0, count, self.image_batch_size):
            pixels = ((rgb[start:start+self.image_batch_size].float()/255 - self.mean)/self.std).to(dtype)
            hidden = (self._tuned_forward(pixels) if self.tune_blocks else
                      self.model(pixel_values=pixels).last_hidden_state)
            patches = hidden[:, prefix:]
            if patches.shape[1:] != (height//16 * (width//16), self.feature_dim):
                raise ValueError('DINOv3 patch grid disagrees with calibrated image dimensions')
            outputs.append(patches)
        return torch.cat(outputs, dim=0)

    def encode(self, rgb):
        """uint8 [B,T,N,3,H,W] -> dict(features=[B,T,N,P,C] patch features, patch_grid=(H/16, W/16))."""
        if rgb.ndim != 6 or rgb.dtype != torch.uint8:
            raise ValueError('Expected uint8 [B,T,N,3,H,W]')
        b, t, n, c, h, w = rgb.shape
        if min(b, t, n) < 1:
            raise ValueError('Empty image group')
        features = self.extract_images(rgb.reshape(b*t*n, c, h, w))
        return dict(features=features.reshape(b, t, n, -1, self.feature_dim), patch_grid=(h//16, w//16))

    def writer_input(self, encoded):
        """Output of ``encode`` -> the writer's input (FP32 patch features and their grid)."""
        return dict(features=encoded['features'].float(), patch_grid=encoded['patch_grid'])

    def forward(self, rgb):
        return self.writer_input(self.encode(rgb))
