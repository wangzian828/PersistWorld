"""Execution options: memory/speed choices that do not change the model function.

Defaults are the settings of all runs in the paper. Triton kernels are used on CUDA only; on CPU
the PyTorch reference paths run instead.
"""
from . import host_sync

DEFAULTS = dict(
    writer_query_chunk=32768,              # sub-cells per writer chunk
    writer_activation_target_gib=32.,      # recompute writer chunks in backward above this allocation
    checkpoint_evolver=False,              # activation checkpointing (also enabled on an out-of-memory retry)
    checkpoint_writer=False,
    checkpoint_reader=False,
    fused_writer=True,                     # Triton evidence sampling and attention (kernels/fused_evidence.py)
    fused_loss=True,                       # Triton focal loss (kernels/focal_loss.py)
    planned_sampling=True,                 # Triton trilinear resampling of the reader logits without gradient
    sync_free_forward=True,                # no host/device round trips in the forward (host_sync.py)
    motion_reduction_chunk=1024,           # parallel partial sums of the motion attention backward
    retain_spatial_probabilities=True,     # keep FP32 attention probabilities of the axial attention for backward
    score_chunk_elements=16*1024**2,       # bound of the attention score temporaries
)


def configure_execution(model, options=None):
    """Apply ``options`` (overrides of DEFAULTS) to the model's modules; returns the full option set."""
    options = {**DEFAULTS, **(options or {})}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown execution options: {sorted(unknown)}')
    if type(options['writer_query_chunk']) is not int or options['writer_query_chunk'] <= 0:
        raise ValueError('writer_query_chunk must be a positive integer')
    if not 0 <= options['writer_activation_target_gib'] <= 60:
        raise ValueError('writer_activation_target_gib must be in [0, 60]')
    writer, evolver = model.writer, model.evolver
    writer.query_chunk = options['writer_query_chunk']
    writer.checkpoint_chunks = options['checkpoint_writer']
    if hasattr(writer, 'fused_evidence'):
        writer.fused_evidence = options['fused_writer']
        writer.activation_target_bytes = int(options['writer_activation_target_gib']*1024**3)
    evolver.checkpoint_blocks = options['checkpoint_evolver']
    model.checkpoint_readout = options['checkpoint_reader']
    model.reader.planned_sampling = options['planned_sampling']
    model.fused_loss = options['fused_loss']
    host_sync.ENABLED = options['sync_free_forward']
    for block in evolver.blocks:
        block.motion.query_reduction_chunk = options['motion_reduction_chunk']
        block.spatial.score_chunk_elements = options['score_chunk_elements']
        block.spatial.retain_fp32_probabilities = options['retain_spatial_probabilities']
    for block in evolver.context_blocks:
        block.spatial.score_chunk_elements = options['score_chunk_elements']
        block.spatial.retain_fp32_probabilities = options['retain_spatial_probabilities']
    model.execution_options = options
    return options
