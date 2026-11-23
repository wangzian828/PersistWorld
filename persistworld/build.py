"""Model construction and checkpoint loading."""
import torch

from .losses import objective_spec
from .model import PersistWorld, PersistWorldConfig
from .model.execution import configure_execution
from .paths import data_path


def build_encoder(cfg, verify_checksums=True):
    """The image encoder of a camera config, or the label pass-through of an occupancy-input config."""
    if cfg['input'] != 'camera':
        from .model.occupancy_writer import OccupancyInput
        return OccupancyInput(cfg['model']['num_classes'])
    from .model.encoder import DINOv3ImageEncoder
    e = cfg['encoder']
    return DINOv3ImageEncoder(data_path(e['checkpoint']), weights_sha256=e['weights_sha256'],
                              config_sha256=e['config_sha256'], image_batch_size=e['image_batch_size'],
                              dtype=torch.bfloat16, kind=e['kind'], tune_blocks=e['tune_blocks'],
                              verify_checksums=verify_checksums)


def build_model(cfg, verify_checksums=True):
    """PersistWorld of a resolved config (``config.load_config``), with its objective and ablation switches.

    Modules are created in the same order and with the same seeds as for the released models, so a given
    global seed reproduces their initialization (camera models).
    """
    encoder = build_encoder(cfg, verify_checksums)
    model = PersistWorld(encoder, PersistWorldConfig(**{k: tuple(v) if isinstance(v, list) else v
                                                         for k, v in cfg['model'].items()}),
                         writer='camera' if cfg['input'] == 'camera' else 'occupancy')
    configure_execution(model, cfg.get('execution'))
    o = cfg['objective']
    m = cfg['model']
    model.occupancy_objective = objective_spec(o['class_priors'], o['free_weight'], o['loss_scale'],
                                               o['current_weight'], m['num_classes'], m['free_id'])
    ablation = cfg.get('ablation')
    if ablation == 'mean_writing':
        model.writer.ablate_attention = True
    elif ablation == 'no_context':
        model.ablate_shared_context = True
    elif ablation == 'no_depth':
        model.writer.ablate_depth = True
        # The geometry embeddings stay in the checkpoint but are neither used nor trained; the ray head still trains.
        for module in (model.writer.geometry_key, model.writer.geometry_value, model.writer.depth_scale):
            module.requires_grad_(False)
    elif ablation == 'ego_state':
        model.ego_state = True
    elif ablation == 'no_evolver':
        model.ablate_evolver = True
    elif ablation is not None:
        raise ValueError(f'Unknown ablation {ablation!r}')
    return model


def trainable_state(model):
    """State dict without the frozen encoder weights (they come from the pretrained checkpoint)."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v.detach().cpu() for n, v in model.state_dict().items() if n in keep or not n.startswith('encoder.')}


def load_weights(model, path):
    """Load a checkpoint written by tools/train.py or tools/train_windowed.py, or a released one."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    result = model.load_state_dict(saved['model'], strict=False)
    if result.unexpected_keys or not all(n.startswith('encoder.') for n in result.missing_keys):
        raise ValueError(f'Checkpoint does not match the model: {result}')
    return saved
