# PersistWorld

**Write, Evolve, Read: Streaming Cameras into a Persistent World State for 4D Occupancy Forecasting**

ZiAn Wang<sup>1</sup>, Tianyi Wang<sup>2</sup>, Zhihao Chen<sup>1</sup>, Manni Cui<sup>3</sup>, Junkai Chen<sup>4</sup>,
ZhiJin Chen<sup>4</sup>, Ruiqi Liu<sup>4</sup>, Junfei Wu<sup>4</sup>, Huanyu Zhang<sup>4</sup>, Shu Wu<sup>4,†</sup>

<sup>1</sup>Jilin University, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>Huazhong University of Science and
Technology, <sup>4</sup>Institute of Automation, Chinese Academy of Sciences. <sup>†</sup>Corresponding author.

[Project page](https://wangzian828.github.io/PersistWorld) · [Weights](https://huggingface.co/ZiAnwang/PersistWorld) · [Paper](https://arxiv.org/abs/2611.30573)

PersistWorld streams camera frames into a 3D world state anchored in the scene. When a capture group of
surround-view images arrives, the **Evolver** carries the state to its capture time, the **Writer** writes the
evidence of the new images into it, and the **Reader** decodes the current occupancy. To forecast, the Evolver
advances a copy of the state to any queried time along the ego trajectory and the Reader decodes it in the queried
ego frame. One model serves every horizon from 1 to 8 s.

On Occ3D-nuScenes it reaches 28.72 mIoU / 67.82 IoU averaged over 1–3 s and 23.86 / 64.73 over 1–8 s, at
23.15 FPS on one A800.

## Code map

| Paper | Code |
|---|---|
| Write, evolve, read: one step per capture group, and the forecasts | `persistworld/model/persistworld.py` |
| World state on a scene-anchored lattice (Sec. 3.1) | `persistworld/model/state.py`, `model/memory.py` |
| Writer and ray-depth sections (Sec. 3.2) | `model/writer.py`, `model/ray_depth.py`, `model/encoder.py` |
| Evolver and shared observed context (Sec. 3.3) | `model/evolver.py`, `model/attention.py` |
| Reader (Sec. 3.4) | `model/reader.py` |
| Unified occupancy objective (Sec. 3.5) | `persistworld/losses.py` |
| Memory span and training on streams (Sec. 3.6) | `persistworld/streaming.py`, `training.py`, `plan.py`, `tools/train.py` |
| Writing from occupancy, -GT / -STC (Appendix B) | `model/occupancy_writer.py` |
| Evaluation protocols | `persistworld/eval_protocol.py`, `tools/evaluate.py` |
| Model variants and their construction | `persistworld/config.py`, `persistworld/build.py`, `configs/` |
| Speed | `persistworld/fast/`, `tools/benchmark_fps.py` |

`persistworld/kernels/` holds Triton kernels used on CUDA (writer evidence, focal loss, modulated norm,
trilinear resampling in the Reader). Each has a PyTorch reference path that computes the same function and is
used on CPU.

## Installation

Python 3.11 and an NVIDIA GPU; we trained on six A800 80 GB GPUs.

```bash
git clone https://github.com/wangzian828/PersistWorld.git && cd PersistWorld
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # or the build for your CUDA
pip install -e .
```

## Data

Data paths are relative to a data root, `./data` by default or `$PERSISTWORLD_DATA`:

```
data/
  occ3d_nuscenes/
    raw_nuscenes/extracted/v1.0-trainval/*.json      nuScenes v1.0-trainval metadata
    raw_nuscenes/extracted/samples/CAM_*/            keyframe images
    raw_nuscenes/extracted/sweeps/CAM_*/             sweep images (only for the windowed model of Tab. 3)
    extracted/gts/<scene>/<token>/labels.npz         Occ3D-nuScenes labels (semantics, mask_camera)
    official_splits.json                             written by tools/prepare_occ3d.py
  models/dinov3-vitb16-pretrain-lvd1689m/            config.json, model.safetensors
  stcocc/pred_occ3d/<scene>/<token>/labels.npz       only for the stc variant
```

1. Download nuScenes v1.0-trainval (metadata and keyframe camera images; sweep images only for the windowed
   model; no LiDAR is read) and the Occ3D-nuScenes ground truth (`gts`), and extract them as above.
2. Download the DINOv3 ViT-B/16 weights `facebook/dinov3-vitb16-pretrain-lvd1689m` from Hugging Face into
   `data/models/dinov3-vitb16-pretrain-lvd1689m`. Their SHA-256 is checked against `configs/persistworld_occ3d.json`.
3. Write the official scene split (700 train / 150 val scenes):
   `pip install nuscenes-devkit && python tools/prepare_occ3d.py`.
4. Only for the `stc` variant: run the released STCOcc model (ResNet-50, Occ3D) on every keyframe and store its
   grid as `semantics` (uint8, 200×200×16) in `data/stcocc/pred_occ3d/<scene>/<token>/labels.npz`.

The class priors of the objective are in the config; `tools/compute_priors.py` recomputes them.

## Pretrained weights

| Model | 1–3 s mIoU / IoU | 1–8 s mIoU / IoU | File |
|---|---|---|---|
| PersistWorld (camera) | 28.72 / 67.82 | 23.86 / 64.73 | `persistworld_main.pt` |

Download [`persistworld_main.pt`](https://huggingface.co/ZiAnwang/PersistWorld) from Hugging Face (also on the [GitHub release](https://github.com/wangzian828/PersistWorld/releases/download/v1.0.0/persistworld_main.pt)) into `checkpoints/`:

```bash
huggingface-cli download ZiAnwang/PersistWorld persistworld_main.pt --local-dir checkpoints
```

A checkpoint holds the trainable parameters only;
the frozen DINOv3 blocks come from the pretrained weights above.

## Evaluation

Streaming evaluation on the val split: one state per scene, reset at scene starts and after every eight capture
groups. A full run takes about 45 minutes on one A800; `--shard i/N` splits the scenes over processes and
`merge` adds up their confusion matrices:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python tools/evaluate.py eval --ckpt checkpoints/persistworld_main.pt \
      --shard $i/4 --out results/main.shard$i.json &
done; wait
python tools/evaluate.py merge --out results/main.json results/main.shard*.json   # prints the averages
```

```bash
# Tab. 1 and Tab. 2: 6019 anchors, camera mask of the target frame, 1-8 s (prints the 1-3 s and 1-8 s averages)
python tools/evaluate.py eval --ckpt checkpoints/persistworld_main.pt --horizons 1,2,3,4,5,6,7,8 --out results/main.json

# Memory span (Fig. 6): reset period N
python tools/evaluate.py eval --ckpt checkpoints/persistworld_main.pt --period 4 --out results/main_N4.json

# Occ-GT block of Tab. 1 and Appendix B: 4519 anchors with 2 s of history and 3 s of future, full volume
python tools/evaluate.py eval --variant gt --ckpt runs/gt/epoch_20.pt --horizons 1,2,3 --out results/gt.json
python tools/evaluate.py eval --variant gt --ckpt runs/gt/epoch_20.pt --protocol come8 --out results/gt_come8.json
# "+ warp" rows of Appendix B: the input occupancy moved to the target frame with the ego poses
python tools/evaluate.py eval --variant gt --baseline copy_forward --horizons 1,2,3 --out results/gt_warp.json

# Tab. 3, windowed model
python tools/evaluate.py windowed --ckpt runs/windowed/epoch_20.pt --horizons 1,2,3 --out results/windowed.json
```

Every other variant is evaluated like the main model with `--variant <name>` and its checkpoint.

## Training

All models of the paper are trained from scratch for 20 epochs on 6 GPUs (about 17 h for the main model on A800):

```bash
torchrun --nproc_per_node=6 tools/train.py --variant main --out runs/main
```

AdamW at 3e-4 (fine-tuned DINOv3 blocks at a third of it), no weight decay, a warmup of 4 updates, cosine decay to a
third over epochs 17–20, gradient clipping at 5. Every update holds 64 supervised keyframes from 18 segments of up
to eight capture groups, with state depths drawn uniformly from 1 to 8 and gradients truncated to the last two
writes. The update plan is fixed for 6 GPUs, so fewer GPUs train the same updates, more slowly. A run resumes from
`OUT/last.pt` when restarted.

| `--variant` | Paper |
|---|---|
| `main` | PersistWorld (Tab. 1, Tab. 2) |
| `abl_ego_state` | Tab. 3, Ego-centric state |
| `abl_frozen_encoder` | Tab. 3, Frozen encoder |
| `abl_no_evolver` | Tab. 3, No Evolver |
| `stc` | Tab. 3, From STCOcc; Appendix B, PersistWorld-STC |
| `gt` | Tab. 1 Occ-GT block; Appendix B, PersistWorld-GT |
| `abl_no_depth` | Appendix C, No depth conditioning |
| `abl_mean_writing` | Appendix C, Mean writing |
| `abl_no_context` | Appendix C, No shared context |

The windowed model of Tab. 3, which rebuilds the state from the last eight capture groups for every query:

```bash
torchrun --nproc_per_node=6 tools/train_windowed.py --out runs/windowed
```

## Speed

```bash
python tools/benchmark_fps.py --ckpt checkpoints/persistworld_main.pt --out results/fps.json                     # paper setting
python tools/benchmark_fps.py --ckpt checkpoints/persistworld_main.pt --out results/fps_ref.json --path reference
```

Batch size one, from the inputs on the GPU to the grids of the current frame and of 1, 2, and 3 s on the CPU. The
default `fastest` configuration (`persistworld/fast/`) uses inference-only kernels; two of its switches are not
bitwise exact, and on the full val split they change the 1–3 s scores by less than 0.01. The paper reports
23.15 FPS on one A800; timings depend on the machine.

## Tests

```bash
python tests/test_cpu_smoke.py   # CPU, no data: streaming steps, loss and backward for both writers
```

## Citation

```bibtex
@article{wang2026persistworld,
  title   = {Write, Evolve, Read: Streaming Cameras into a Persistent World State for 4D Occupancy Forecasting},
  author  = {Wang, ZiAn and Wang, Tianyi and Chen, Zhihao and Cui, Manni and Chen, Junkai and Chen, ZhiJin
             and Liu, Ruiqi and Wu, Junfei and Zhang, Huanyu and Wu, Shu},
  journal = {arXiv preprint arXiv:2611.30573},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The pretrained weights are trained on nuScenes and
Occ3D-nuScenes and are released for non-commercial research use, following the nuScenes terms of use
(CC BY-NC-SA 4.0); the DINOv3 backbone weights are subject to their own license.
