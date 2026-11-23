"""Execution-only inference switches for the latency benchmark (single-state streaming, batch size one).

Same model and inputs; current and 1/2/3 s grids as outputs. ``install(model, fastest=True)`` enables the
configuration of the paper's FPS (one A800); ``fastest=False`` leaves out the two switches that are not
bitwise exact (``motion_fold``, ``enc_compile``). All switches need CUDA.

Base switches (always on):
* the encoder of each capture group runs as a CUDA graph (``encode_group``);
* the evolver's attention runs through BF16 SDPA kernels (FP32 softmax inside) instead of the chunked FP32
  math attention used for training; 4x4 pose solves via cuSOLVER;
* torch.compile (dynamic shapes) of the evolver/context blocks, the reader and the writer chunk; the writer
  chunk uses the fused inference kernels of fused_writer.py, 131072 sub-cells per chunk;
* a first write into an empty session advances by zero time, so the evolver is skipped (exact);
* the observed context built for the forecasts of a keyframe is reused for the evolver prior of the next
  keyframe (the state has not changed in between; exact).

Switches of ``configure`` (exact unless noted):
* ``stamp_cache``     the parameter stamp that validates a reused context is computed once per model;
* ``dense_bank``      memory tiles live in one preallocated GPU bank per field (index_select / index_copy);
* ``precast``         Linear/Conv weights of the evolver and the reader stored in BF16 (autocast casts them anyway);
* ``readout_argmax``  the reader's trilinear resampling and the class argmax in one Triton kernel (labels out);
* ``short_attn``      the evolver's z-axis self-attention (lines of <= 16 cells) in a Triton kernel with the
                      rounding steps of FlashAttention-2 (bitwise identical outputs);
* ``fused_rope``      DINOv3's rotary embedding in one Triton kernel per tensor (bitwise identical);
* ``enc_overlap``     the encoder runs on a side stream while the evolver advances the state;
* ``motion_fold``     motion cross-attention with W_q folded into the keys and W_o into the values (not bitwise);
* ``enc_compile``     the encoder compiled by Inductor and replayed from a CUDA graph (not bitwise).
"""
import types

import numpy as np
import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..model import attention
from ..model import evolver as evolver_lib
from ..model import memory
from ..model import reader as reader_lib

FLAGS = dict(stamp_cache=False, dense_bank=False, precast=False, precast_bias=False, readout_argmax=False,
             enc_overlap=False, fused_rope=False, short_attn=False, motion_fold=False, enc_compile=False)
EXACT = dict(stamp_cache=True, dense_bank=True, precast=True, readout_argmax=True, enc_overlap=True, fused_rope=True,
             short_attn=True)
FASTEST = dict(EXACT, motion_fold=True, enc_compile=True)
_ORIGINAL = {}
_STAMPS = {}


# ------------------------------------------------------------------------------------------ encoder
class GraphEncoder:
    """The eval DINOv3 group encoder replayed from a CUDA graph (same kernels, same values)."""
    def __init__(self, encoder, shape):
        self.input = torch.zeros(shape, dtype=torch.uint8, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(3):
                encoder.encode(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.output = encoder.encode(self.input)['features']

    def __call__(self, rgb):
        self.input.copy_(rgb, non_blocking=True)
        self.graph.replay()
        return self.output.clone()


_GRAPH_SETS = {}
_ENCODERS = {}
_ENC_PENDING = {}
_SIDE = []


def _encode(m, rgb):
    """The group encoder of the current configuration: compiled, or the eager CUDA graph (one per RoPE setting)."""
    key = tuple(rgb.shape)
    if FLAGS['enc_compile']:
        enc = _ENCODERS.get(key)
        if enc is None:
            enc = _ENCODERS[key] = CompiledEncoder(m.encoder, key, None)
        return enc(rgb)
    graphs = _GRAPH_SETS.setdefault(bool(FLAGS['fused_rope']), {})
    graph = graphs.get(key)
    if graph is None:
        graph = graphs[key] = GraphEncoder(m.encoder, key)
    return graph(rgb)


def encode_group(m, rgb):
    """[1,1,6,3,H,W] uint8 -> encoder features [1,1,6,P,C]; on a side stream with ``enc_overlap``."""
    if not FLAGS['enc_overlap']:
        return _encode(m, rgb)
    if not _SIDE:
        _SIDE.append(torch.cuda.Stream())
    side = _SIDE[0]
    main = torch.cuda.current_stream()
    side.wait_stream(main)                       # the group's images are on the GPU
    with torch.cuda.stream(side):
        out = _encode(m, rgb)
    event = torch.cuda.Event()
    event.record(side)
    out.record_stream(main)                      # made on the side stream, read on the main one
    if rgb.is_cuda:
        rgb.record_stream(side)
    _ENC_PENDING['event'] = event
    return out


def _wait_for_encoder(writer_input):
    def wait_then_writer_input(encoded, *args, **kw):
        event = _ENC_PENDING.pop('event', None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        return writer_input(encoded, *args, **kw)
    return wait_then_writer_input


# ------------------------------------------------------------------------------------------ attention
def sdpa_multihead(module, query, memory, chunk=None, attn_mask=None):
    """Inference twin of attention.bounded_math_multihead through BF16 SDPA kernels (FP32 softmax inside)."""
    width = module.embed_dim
    b, nq, _ = query.shape
    nk = memory.shape[1]
    heads = module.num_heads
    hd = width//heads
    if query is memory:
        q, k, v = F.linear(query, module.in_proj_weight, module.in_proj_bias).view(b, nq, 3, heads, hd).unbind(2)
    else:
        qw, kvw = module.in_proj_weight.split([width, 2*width])
        qb, kvb = (None, None) if module.in_proj_bias is None else module.in_proj_bias.split([width, 2*width])
        q = F.linear(query, qw, qb).view(b, nq, heads, hd)
        k, v = F.linear(memory, kvw, kvb).view(b, nk, 2, heads, hd).unbind(2)
    dtype = q.dtype
    mask = None
    if attn_mask is not None:
        mask = ~attn_mask if attn_mask.dtype == torch.bool else attn_mask.float()
        mask = mask[None, None] if mask.ndim == 2 else mask[:, None]
    with torch.autocast('cuda', enabled=False):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    out = out.to(dtype).transpose(1, 2).reshape(b, nq, width)
    return F.linear(out, module.out_proj.weight, module.out_proj.bias)


# ------------------------------------------------------------------------------------------ stamp cache
def _parameter_stamp(model):
    if FLAGS['stamp_cache'] and not model.training and not torch.is_grad_enabled():
        value = _STAMPS.get(id(model))
        if value is None:
            value = _STAMPS[id(model)] = _ORIGINAL['parameter_stamp'](model)
        return value
    return _ORIGINAL['parameter_stamp'](model)


# ------------------------------------------------------------------------------------------ dense bank
class DenseTile:
    """Stand-in for memory.MemoryTile whose tensors are views of the session's dense bank."""
    __slots__ = ('bank', 'slot')

    def __init__(self, bank, slot):
        self.bank = bank
        self.slot = slot

    def _view(self, field):
        b = self.bank
        lo = self.slot*b.volume
        x = getattr(b, field)[lo:lo+b.volume]
        return x.view(*b.tile_shape, b.dim) if field == 'features' else x.view(b.tile_shape)

    features = property(lambda self: self._view('features'))
    state_time = property(lambda self: self._view('state_time'))
    observed_time = property(lambda self: self._view('observed_time'))
    valid = property(lambda self: self._view('valid'))

    def to(self, device, detach=False):
        if torch.device(device) != self.bank.features.device:
            raise ValueError('Dense bank tiles stay on their GPU')
        return self


class DenseBank:
    """All tiles of one SceneMemory: features [slots*V, D], state_time/observed_time/valid [slots*V]."""
    def __init__(self, tile_shape, dim, dtype, device, capacity=512):
        self.tile_shape = tuple(tile_shape)
        self.dim = dim
        self.volume = int(np.prod(tile_shape))
        self.slot = {}
        self.free = []
        self.capacity = 0
        self.features = torch.zeros((0, dim), dtype=dtype, device=device)
        self.state_time = torch.zeros(0, device=device)
        self.observed_time = torch.zeros(0, device=device)
        self.valid = torch.zeros(0, dtype=torch.bool, device=device)
        self._grow(capacity)
        self.strides = (self.tile_shape[1]*self.tile_shape[2], self.tile_shape[2], 1)

    def _grow(self, capacity):
        old = self.capacity
        extra = capacity-old
        V = self.volume
        dev = self.features.device
        self.features = torch.cat([self.features,
                                   torch.zeros((extra*V, self.dim), dtype=self.features.dtype, device=dev)])
        self.state_time = torch.cat([self.state_time, torch.zeros(extra*V, device=dev)])
        self.observed_time = torch.cat([self.observed_time, torch.full((extra*V,), -torch.inf, device=dev)])
        self.valid = torch.cat([self.valid, torch.zeros(extra*V, dtype=torch.bool, device=dev)])
        self.free.extend(range(capacity-1, old-1, -1))
        self.capacity = capacity

    def collect(self, tiles):
        """Free the slots of tiles the stream has dropped (Stream.finish pops keys from memory.tiles)."""
        if len(self.slot) == len(tiles) and all(isinstance(t, DenseTile) and t.bank is self for t in tiles.values()):
            return []
        stale = [k for k in self.slot if k not in tiles]
        freed = [self.slot.pop(k) for k in stale]
        self.free.extend(freed)
        return freed

    def reset_slots(self, slots):
        """Defaults of a new tile (memory.write_window): features 0, times 0 / -inf, invalid."""
        if not slots:
            return
        V = self.volume
        cells = (np.asarray(slots, np.int64)[:, None]*V+np.arange(V, dtype=np.int64)[None]).reshape(-1)
        from ..model import host_sync as hs
        idx = hs.to_device(torch.from_numpy(cells), self.features.device)
        self.features.index_fill_(0, idx, 0.)
        self.state_time.index_fill_(0, idx, 0.)
        self.observed_time.index_fill_(0, idx, -torch.inf)
        self.valid.index_fill_(0, idx, False)

    def addresses(self, origin, shape, tiles, allocate):
        """Flat bank address of every window cell (-1: no stored tile), and the overlapping keys in
        memory.SceneMemory._overlap order (itertools.product over the key ranges)."""
        t = self.tile_shape
        k0 = [o//s for o, s in zip(origin, t)]
        k1 = [(o+n-1)//s for o, n, s in zip(origin, shape, t)]
        kshape = [b-a+1 for a, b in zip(k0, k1)]
        slots = np.full(kshape, -1, np.int64)
        keys = []
        new = []
        for i in range(kshape[0]):
            for j in range(kshape[1]):
                for k in range(kshape[2]):
                    key = (k0[0]+i, k0[1]+j, k0[2]+k)
                    s = self.slot.get(key)
                    if s is None and key in tiles:
                        raise ValueError('Memory tile outside the dense bank')
                    if s is None and allocate:
                        if not self.free:
                            self._grow(2*self.capacity)
                        s = self.slot[key] = self.free.pop()
                        new.append(s)
                    if s is not None:
                        slots[i, j, k] = s
                        keys.append(key)
        g = [np.arange(n, dtype=np.int64)+o for o, n in zip(origin, shape)]
        kk = [gi//s-a for gi, s, a in zip(g, t, k0)]
        ll = [gi % s for gi, s in zip(g, t)]
        slot = slots[kk[0][:, None, None], kk[1][None, :, None], kk[2][None, None, :]]
        local = (ll[0][:, None, None]*self.strides[0]+ll[1][None, :, None]*self.strides[1]+ll[2][None, None, :])
        index = np.where(slot >= 0, slot*self.volume+local, -1)
        return index.reshape(-1), keys, new


def _dense_of(bank, create, dtype=None, device=None):
    dense = getattr(bank, '_dense', None)
    if dense is None and create and not bank.tiles:
        dense = bank._dense = DenseBank(bank.tile_shape, bank.dim, dtype, device)
    return dense


def _read_window(bank, origin, initial, default_time):
    dense = getattr(bank, '_dense', None)
    if dense is None:
        return _ORIGINAL['read_window'](bank, origin, initial, default_time)
    from ..model.memory import WorldWindow
    from ..model import host_sync as hs
    if torch.is_grad_enabled():
        raise ValueError('Dense bank is inference-only')
    shape = tuple(initial.shape[1:4])
    device = initial.device
    dense.collect(bank.tiles)   # freed slots are reset when they are allocated again
    index, keys, _ = dense.addresses(tuple(origin), shape, bank.tiles, False)
    times = torch.full(shape, float(default_time), device=device, dtype=torch.float32)
    observed = torch.full_like(times, -torch.inf)
    if not keys:
        return WorldWindow(initial, tuple(origin), bank.cell_size, times[None], observed[None],
                           torch.zeros((1, *shape), device=device, dtype=torch.bool))
    index = hs.to_device(torch.from_numpy(index), device)
    selected = index.clamp_min(0)
    valid = dense.valid.index_select(0, selected).reshape(shape) & (index >= 0).reshape(shape)
    features = dense.features.index_select(0, selected).reshape(*shape, dense.dim)
    values = torch.where(valid[..., None], features.to(initial.dtype), initial[0])
    times = torch.where(valid, dense.state_time.index_select(0, selected).reshape(shape), times)
    observed = torch.where(valid, dense.observed_time.index_select(0, selected).reshape(shape), observed)
    return WorldWindow(values[None], tuple(origin), bank.cell_size, times[None], observed[None], valid[None])


def _write_window(bank, window, observed_time=None):
    use = FLAGS['dense_bank'] and not torch.is_grad_enabled() and window.features.is_cuda
    dense = _dense_of(bank, use, window.features.dtype, window.features.device)
    if dense is None:
        return _ORIGINAL['write_window'](bank, window, observed_time)
    from ..model import host_sync as hs
    if torch.is_grad_enabled():
        raise ValueError('Dense bank is inference-only')
    if window.features.dtype != dense.features.dtype:
        raise ValueError('Packed writes require consistent feature dtype')
    dense.collect(bank.tiles)
    index, keys, new = dense.addresses(tuple(window.origin), tuple(window.shape), bank.tiles, True)
    dense.reset_slots(new)
    index = hs.to_device(torch.from_numpy(index), window.features.device)
    observed = window.observed_time if observed_time is None else observed_time
    dense.features.index_copy_(0, index, window.features[0].reshape(-1, dense.dim))
    dense.state_time.index_copy_(0, index, window.state_time[0].reshape(-1))
    dense.observed_time.index_copy_(0, index, observed[0].reshape(-1))
    dense.valid.index_fill_(0, index, True)
    for key in keys:   # every overlapping tile is replaced (copy-on-write identity, as write_window)
        bank.tiles[key] = DenseTile(dense, dense.slot[key])
    bank.revision += 1
    return set(keys)


def _fork(self):
    if getattr(self, '_dense', None) is not None:
        raise ValueError('Dense-bank memories are not forked (inference streams only)')
    return _ORIGINAL['fork'](self)


_PRECAST = []   # (module, attribute, FP32 parameter, BF16 copy, is_bias)


def precast_registry(model):
    """BF16 copies of the Linear/Conv/attention weights autocast casts at every call (evolver and reader)."""
    from torch import nn
    if _PRECAST:
        return _PRECAST
    for root in (model.evolver, model.reader):
        for name, module in root.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                tensors = ['weight', 'bias']
            elif isinstance(module, nn.MultiheadAttention):
                tensors = ['in_proj_weight', 'in_proj_bias']
            else:
                continue
            for t in tensors:
                p = getattr(module, t, None)
                if p is None or p.dtype != torch.float32:
                    continue
                _PRECAST.append((module, t, p, nn.Parameter(p.detach().to(torch.bfloat16), requires_grad=False),
                                 t.endswith('bias')))
    return _PRECAST


def _apply_precast(weights, biases):
    """Point the registered modules at their BF16 copies (or back at the FP32 parameters)."""
    for module, t, full, half, bias in _PRECAST:
        want = half if (biases if bias else weights) else full
        if getattr(module, t) is not want:
            setattr(module, t, want)


@triton.jit(do_not_specialize=['N'])
def _gather_argmax(Value, Indices, Weights, Out, N, C: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
                   ROUND_BF16: tl.constexpr):
    """kernels/planned_sampling._gather (same corner order, same FMA accumulation) + argmax over the C classes."""
    p = tl.program_id(0)*P+tl.arange(0, P)
    c = tl.arange(0, K)
    result = tl.full((P, K), 0, tl.float32)
    for corner in tl.static_range(8):
        idx = tl.load(Indices+corner*N+p, p < N, 0)
        w = tl.load(Weights+corner*N+p, p < N, 0)
        v = tl.load(Value+idx[:, None]*C+c[None, :], (p[:, None] < N) & (c[None, :] < C), 0).to(tl.float32)
        result = result+v*w[:, None]
    if ROUND_BF16:   # the stored sample is BF16; argmax compares the rounded values
        result = result.to(tl.bfloat16).to(tl.float32)
    result = tl.where(c[None, :] < C, result, float('-inf'))
    best = tl.max(result, 1)
    label = tl.min(tl.where(result == best[:, None], c[None, :], K), 1)   # first maximal class
    tl.store(Out+p, label.to(tl.uint8), p < N)


def planned_argmax(value, grid):
    """Class labels (uint8) of ``planned_grid_sample(value, grid)`` without materializing the sampled logits."""
    from ..kernels.planned_sampling import _plan
    if (value.ndim != 5 or value.shape[0] != 1 or grid.dtype != torch.float32
            or value.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('Expected one [1,X,Y,Z,C] BF16/FP32 volume and FP32 coordinates')
    value = value.contiguous()
    grid = grid.contiguous()
    _, x, y, z, c = value.shape
    n = grid.numel()//3
    indices = torch.empty((8, n), device=value.device, dtype=torch.int32)
    weights = torch.empty((8, n), device=value.device, dtype=torch.float32)
    out = torch.empty((1, *grid.shape[:-1]), device=value.device, dtype=torch.uint8)
    _plan[(triton.cdiv(n, 256),)](grid, indices, weights, n, x, y, z, 256, enable_fp_fusion=True)
    _gather_argmax[(triton.cdiv(n, 32),)](value, indices, weights, out, n, c, 32, triton.next_power_of_2(c),
                                          value.dtype == torch.bfloat16, enable_fp_fusion=True)
    return out


def _read_world_query(readout, query, output_shape, bounds):
    """reader.read_world_query returning class labels (``readout_argmax``)."""
    if not (FLAGS['readout_argmax'] and not torch.is_grad_enabled() and getattr(readout, 'planned_sampling', False)):
        return _ORIGINAL['read_world_query'](readout, query, output_shape, bounds)
    from ..model import host_sync as hs
    from ..model import geometry
    from ..model.state import reference_grid
    if query.features.shape[0] != 1:
        raise ValueError('World readout requests retain one scene origin each')
    logits = readout(query.features)
    with torch.autocast(device_type=logits.device.type, enabled=False):   # = reader.read_world_query
        xyz = reference_grid(output_shape, bounds, device=logits.device)
        points = geometry.transform_points(query.scene_from_ego, xyz.reshape(-1, 3))
        lower = hs.upload(list(query.origin), xyz)*hs.const(list(query.cell_size), xyz)
        span = hs.const(list(query.features.shape[1:4]), xyz)*hs.const(list(query.cell_size), xyz)
        grid = (2*(points-lower)/span-1).reshape(*output_shape, 3)
        return planned_argmax(logits, grid)


@triton.jit
def _ex2(x, FTZ: tl.constexpr):
    if FTZ == 1:
        return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True,
                                         pack=1)
    elif FTZ == 2:
        return libdevice.exp2(x)
    else:
        return tl.exp2(x)


@triton.jit
def _fa2_row_sum(p, BL: tl.constexpr, LP: tl.constexpr):
    """FlashAttention-2's order for <= 16 keys (one 16x8x16 MMA tile pair per row): thread t of the quad sums keys
    2t, 2t+1, 8+2t, 9+2t in that order, then the quad all-reduce adds (t0+t2)+(t1+t3)."""
    x = tl.permute(tl.reshape(p, (BL, LP, 2, 4, 2)), (0, 1, 3, 2, 4))   # key = 8a+2t+b -> [.., t, a, b]
    e0, e1 = tl.split(x)                                                  # b = 0 / 1
    e00, e01 = tl.split(e0)                                               # a = 0 / 1
    e10, e11 = tl.split(e1)
    st = ((e00+e10)+e01)+e11                                              # [BL, LP, t]
    st = tl.permute(tl.reshape(st, (BL, LP, 2, 2)), (0, 1, 3, 2))         # t = 2hi+lo -> [.., lo, hi]
    h0, h1 = tl.split(st)
    y0, y1 = tl.split(h0+h1)                                              # lo = 0: s0+s2, lo = 1: s1+s3
    return y0+y1


# Self-attention of BL lines of at most 16 cells (padded to LP) for one head; QKV [B,L,3D] packed as F.linear
# returns it. MASKED: Keep [B,L] marks the cells that are keys. ORDER, NORM, FTZ: see SHORT_VARIANT.
@triton.jit(do_not_specialize=['B', 'L'])
def _short_attn(QKV, Keep, Out, B, L, scale_log2, H: tl.constexpr, HD: tl.constexpr, LP: tl.constexpr, BL: tl.constexpr,
                MASKED: tl.constexpr, ORDER: tl.constexpr, NORM: tl.constexpr, FTZ: tl.constexpr):
    b = tl.program_id(0)*BL+tl.arange(0, BL)
    h = tl.program_id(1)
    i = tl.arange(0, LP)
    e = tl.arange(0, HD)
    D = H*HD
    row = b[:, None]*L+i[None, :]                                           # [BL, LP]
    ok = (b[:, None] < B) & (i[None, :] < L)
    m3 = ok[:, :, None] & (e[None, None, :] < HD)
    base = row[:, :, None]*(3*D)+h*HD+e[None, None, :]
    q = tl.load(QKV+base, m3, 0.)
    k = tl.load(QKV+base+D, m3, 0.)
    v = tl.load(QKV+base+2*D, m3, 0.)
    s = tl.dot(q, tl.permute(k, (0, 2, 1)))                                # [BL, LP, LP] FP32
    if MASKED:
        keep = tl.load(Keep+row, ok, 0) != 0                                # [BL, LP] key kept
        s = tl.where(keep[:, None, :], s, float('-inf'))
    else:
        s = tl.where((i < L)[None, None, :], s, float('-inf'))
    mx = tl.max(s, 2)
    ms = tl.where(mx == float('-inf'), 0., mx*scale_log2)
    # One FFMA, as FlashAttention-2's exp2f(x*scale - max_scaled).
    p = _ex2(s*scale_log2-ms[:, :, None], FTZ)
    if ORDER == 0:
        l = tl.sum(p, 2)
    else:
        l = _fa2_row_sum(p, BL, LP)
    acc = tl.dot(p.to(tl.bfloat16), v)                                     # [BL, LP, HD] FP32
    if NORM == 0:
        o = tl.where((l > 0)[:, :, None], acc/tl.where(l > 0, l, 1.)[:, :, None], 0.)
    else:                                                                  # FA2: acc *= (sum == 0 ? 1 : 1/sum)
        if NORM == 1:
            inv = tl.math.div_rn(tl.full(l.shape, 1., tl.float32), tl.where(l > 0, l, 1.))
        else:
            inv = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [tl.where(l > 0, l, 1.)],
                                            dtype=tl.float32, is_pure=True, pack=1)
        o = acc*inv[:, :, None]
    tl.store(Out+row[:, :, None]*D+h*HD+e[None, None, :], o.to(tl.bfloat16), m3)


# Arithmetic variant 1000*no_contraction + 100*ORDER + 10*NORM + FTZ. 1010 = FlashAttention-2 as PyTorch runs it
# (exp2 argument as a separate multiply and subtract, plain row sum, acc *= IEEE 1/sum). On evolver z-lines
# captured during evaluation, 0 of 117,282,816 outputs differ from SDPA's flash kernel (with FP contraction: 7,364;
# with acc/sum: 26,230).
SHORT_VARIANT = 1010


@torch.library.custom_op('persistworld::short_attention', mutates_args=())
def short_attention(qkv: torch.Tensor, keep: torch.Tensor, heads: int, masked: bool, variant: int) -> torch.Tensor:
    B, L, D3 = qkv.shape
    D = D3//3
    HD = D//heads
    out = torch.empty((B, L, D), device=qkv.device, dtype=qkv.dtype)
    BL = 4
    contract, order, norm, ftz = variant//1000 == 0, variant//100 % 10, variant//10 % 10, variant % 10
    _short_attn[(triton.cdiv(B, BL), heads)](qkv.contiguous(), keep.contiguous(), out, B, L,
                                            HD**-.5*1.4426950408889634, heads, HD, 16, BL, masked, order, norm, ftz,
                                            enable_fp_fusion=contract)
    return out


@short_attention.register_fake
def _(qkv, keep, heads, masked, variant):
    B, L, D3 = qkv.shape
    return qkv.new_empty((B, L, D3//3))


def _short_multihead(module, query, memory, chunk=None, attn_mask=None):
    """Attention of the inference path: ``short_attn`` / ``motion_fold`` where they apply, else sdpa_multihead."""
    from torch.nn import functional as F
    # The evolver's unmasked z-axis lines (<= 16 cells; the reference runs SDPA's flash kernel on them).
    if (FLAGS['short_attn'] and query is memory and attn_mask is None and query.shape[1] <= 16
            and not torch.is_grad_enabled() and query.is_cuda):
        qkv = F.linear(query, module.in_proj_weight, module.in_proj_bias)
        if qkv.dtype == torch.bfloat16 and module.embed_dim % module.num_heads == 0:
            keep = qkv.new_zeros((1,), dtype=torch.uint8)
            out = torch.ops.persistworld.short_attention(qkv, keep, module.num_heads, False, SHORT_VARIANT)
            return F.linear(out, module.out_proj.weight, module.out_proj.bias)
    if (FLAGS['motion_fold'] and query is not memory and memory.shape[1] <= 16 and not torch.is_grad_enabled()
            and query.is_cuda and query.shape[0] == 1 and (attn_mask is None or attn_mask.dtype == torch.bool)):
        return _folded_cross_attention(module, query, memory, attn_mask)
    return _ORIGINAL['multihead'](module, query, memory, chunk=chunk, attn_mask=attn_mask)


def _folded_cross_attention(module, query, memory, attn_mask):
    """MHA(query [1,N,d], keys/values from memory [1,L,d]) with W_q folded into the keys and W_o into the values."""
    from torch.nn import functional as F
    d = module.embed_dim
    H = module.num_heads
    hd = d//H
    b, N, _ = query.shape
    L = memory.shape[1]
    qw, kw, vw = module.in_proj_weight.split([d, d, d])
    qb, kb, vb = module.in_proj_bias.split([d, d, d])
    k = F.linear(memory, kw, kb).float().view(b, L, H, hd)
    v = F.linear(memory, vw, vb).float().view(b, L, H, hd)
    scale = hd**-.5
    with torch.autocast(device_type='cuda', enabled=False):
        kt = torch.einsum('hed,blhe->bdhl', qw.float().view(H, hd, d), k).reshape(b, d, H*L)*scale   # [1, d, H*L]
        sb = torch.einsum('he,blhe->bhl', qb.float().view(H, hd), k).reshape(b, 1, H*L)*scale         # [1, 1, H*L]
        vt = torch.einsum('blhe,dhe->bhld', v, module.out_proj.weight.float().view(d, H, hd)).reshape(b, H*L, d)
        scores = torch.matmul(query.to(torch.bfloat16), kt.to(torch.bfloat16)).float()+sb               # [1, N, H*L]
        scores = scores.view(b, N, H, L)
        if attn_mask is not None:
            excluded = attn_mask if attn_mask.dim() == 3 else attn_mask[None]   # True = not a key
            scores = scores.masked_fill(excluded[:, :, None, :], float('-inf'))
        probs = torch.softmax(scores, -1).reshape(b, N, H*L)
        out = torch.matmul(probs.to(torch.bfloat16), vt.to(torch.bfloat16))
        return (out.float()+module.out_proj.bias.float()).to(torch.bfloat16)


@triton.jit(do_not_specialize=['T', 'P', 'NROWS'])
def _rope(X, Cos, Sin, Out, sb, sh, st, NROWS, T, P, H: tl.constexpr, HD: tl.constexpr, BR: tl.constexpr):
    """X [B,H,T,HD] (strides sb, sh, st, 1) -> Out contiguous; rows t >= T-P get RoPE with Cos/Sin [P,HD]."""
    r = tl.program_id(0)*BR+tl.arange(0, BR)            # flat (b, h, t) rows of the contiguous output
    rm = r < NROWS
    t = r % T
    bh = r//T
    h = bh % H
    b = bh//H
    d = tl.arange(0, HD)
    src = (b*sb+h*sh+t*st)[:, None]
    x = tl.load(X+src+d[None, :], rm[:, None], 0.)
    half = HD//2
    rot_d = tl.where(d < half, d+half, d-half)
    xr = tl.load(X+src+rot_d[None, :], rm[:, None], 0.)
    xr = tl.where((d < half)[None, :], -xr, xr)                              # rotate_half (negation is exact)
    pi = t-(T-P)
    isp = pi >= 0
    c = tl.load(Cos+(tl.where(isp, pi, 0)*HD)[:, None]+d[None, :], (rm & isp)[:, None], 0.)
    sn = tl.load(Sin+(tl.where(isp, pi, 0)*HD)[:, None]+d[None, :], (rm & isp)[:, None], 0.)
    a = (x.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16)
    bb = (xr.to(tl.float32)*sn.to(tl.float32)).to(tl.bfloat16)
    y = (a.to(tl.float32)+bb.to(tl.float32)).to(tl.bfloat16)
    y = tl.where(isp[:, None], y, x)
    tl.store(Out+r[:, None]*HD+d[None, :], y, rm[:, None])


def _rope_one(x, cos, sin):
    """DINOv3's rotary embedding of the patch tokens of one tensor [B,H,T,HD] (the last P positions)."""
    B, H, T, HD = x.shape
    P = cos.shape[-2]
    out = torch.empty((B, H, T, HD), device=x.device, dtype=x.dtype)
    rows = B*H*T
    # No FP contraction: with it Triton fuses each product into an FMA across its BF16 rounding (measured: 23% of
    # the outputs then differ by one BF16 ulp from HF's eager mul/mul/add).
    _rope[(triton.cdiv(rows, 64),)](x, cos.contiguous(), sin.contiguous(), out, x.stride(0), x.stride(1), x.stride(2),
                                    rows, T, P, H, HD, 64, enable_fp_fusion=False)
    return out


def rope_applies(q, k, cos, sin):
    """Whether the fused RoPE kernel supports these inputs (BF16 CUDA, contiguous last dimension)."""
    return (q.dtype == k.dtype == cos.dtype == sin.dtype == torch.bfloat16 and q.is_cuda and q.stride(-1) == 1
            and k.stride(-1) == 1 and cos.dim() == 2 and q.shape[-1] == cos.shape[-1] and q.shape[-1] % 2 == 0)


class CompiledEncoder:
    """Inductor-compiled group encoder (static shapes) captured in a CUDA graph."""
    def __init__(self, encoder, shape, mode):
        self.fn = torch.compile(lambda x: encoder.encode(x)['features'], mode=mode, dynamic=False)
        self.input = torch.zeros(shape, dtype=torch.uint8, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(3):
                self.fn(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.output = self.fn(self.input)

    def __call__(self, rgb):
        self.input.copy_(rgb, non_blocking=True)
        self.graph.replay()
        return self.output.clone()


def _install_short_attention():
    # The function replaced here is sdpa_multihead, installed by _install_base.
    _ORIGINAL['multihead'] = attention.bounded_math_multihead
    attention.bounded_math_multihead = _short_multihead


def _install_fused_rope():
    import transformers.models.dinov3_vit.modeling_dinov3_vit as hf
    original = _ORIGINAL.setdefault('rope', hf.apply_rotary_pos_emb)

    def apply_rotary_pos_emb(q, k, cos, sin, **kwargs):
        if FLAGS['fused_rope'] and not torch.compiler.is_compiling() and rope_applies(q, k, cos, sin):
            return _rope_one(q, cos, sin), _rope_one(k, cos, sin)
        return original(q, k, cos, sin, **kwargs)
    hf.apply_rotary_pos_emb = apply_rotary_pos_emb


def _install_base(model):
    from torch import _dynamo
    from . import fused_writer
    if model.ego_state or model.ablate_shared_context or getattr(model.writer, 'ablate_depth', False):
        raise ValueError('The fast path is for the main camera model; run ablations through tools/evaluate.py')
    for name in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(_dynamo.config, name):
            setattr(_dynamo.config, name, 256)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    attention.bounded_math_multihead = sdpa_multihead
    # The observed context depends only on the committed session and the parameters: the context built for the
    # forecasts of keyframe t is the one the evolver prior of keyframe t+1 needs. The key is the context's own
    # validity check, so a changed session or parameter set always recomputes.
    original_context = model._prepare_forecast_context
    cache = {}

    def _prepare_forecast_context(session):
        key = (id(session.memory), session.memory.revision, model.observation_stamp(session),
               evolver_lib.parameter_stamp(model), model.training, torch.is_grad_enabled())
        if cache.get('key') != key:
            cache['key'] = key
            cache['value'] = original_context(session)
        return cache['value']
    model._prepare_forecast_context = _prepare_forecast_context
    model.context_reuse_cache = cache
    # A first write into an empty session advances by zero time: the evolver returns the window unchanged.
    original_predict = model._predict

    def _predict(self, session, poses, times, context=None):
        if session.updates == 0 and times.shape[1] == 1 and session.memory.tiles == {}:
            window, _ = self._prediction_inputs(session, poses)
            target = times[:, -1]
            # The evolver's torch.where(elapsed == 0, features, FP32 result) promotes the features to FP32.
            return window.replace(window.features.float(),
                                  state_time=target[:, None, None, None].expand_as(window.state_time))
        return original_predict(session, poses, times, context)
    model._predict = types.MethodType(_predict, model)
    for block in list(model.evolver.blocks)+list(model.evolver.context_blocks):
        block.forward = torch.compile(block.forward, dynamic=True)
    model.writer.query_chunk = 131072
    fused_writer.install(model)
    model.writer._chunk = torch.compile(model.writer._chunk, dynamic=True)
    model.reader.forward = torch.compile(model.reader.forward, dynamic=True)


def configure(model, flags):
    """Set every switch (unnamed ones off) and drop state computed under the previous configuration."""
    unknown = set(flags)-set(FLAGS)
    if unknown:
        raise ValueError(f'Unknown switches {unknown}')
    new = {k: bool(flags.get(k, False)) for k in FLAGS}
    _apply_precast(new['precast'], new['precast_bias'])
    FLAGS.update(new)
    _STAMPS.clear()
    model.context_reuse_cache.clear()
    return dict(FLAGS)


def install(model, fastest=True):
    """Install all switches on an eval model on CUDA; ``fastest`` selects the paper's FPS configuration."""
    if not _ORIGINAL:
        _install_base(model)
        model.encoder.writer_input = _wait_for_encoder(model.encoder.writer_input)
        _install_fused_rope()
        _ORIGINAL.update(parameter_stamp=evolver_lib.parameter_stamp, read_window=memory.read_window,
                         write_window=memory.write_window, fork=memory.SceneMemory.fork,
                         read_world_query=reader_lib.read_world_query)
        evolver_lib.parameter_stamp = _parameter_stamp
        memory.read_window = _read_window
        memory.write_window = _write_window
        memory.SceneMemory.fork = _fork
        reader_lib.read_world_query = _read_world_query
        _install_short_attention()
    precast_registry(model)
    return configure(model, FASTEST if fastest else EXACT)
