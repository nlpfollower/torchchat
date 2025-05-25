from os import environ
import types
from typing import Optional

import torch
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torchchat.model import TransformerArgs, ModelType
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.models.llama import TransformerModelArgs as ModelArgs, llama3_configs
from torchtitan.parallelisms import ParallelDims
from torchtitan.models.llama.parallelize_llama import parallelize_llama
from torchtitan.utils import device_type, get_device_info, set_determinism
from torchchat.model import Transformer as TorchchatTransformer, TextOnlyModel, ModelArgs as TorchchatModelArgs

from torchchat.distributed.utils import init_distributed


def convert_model_args_to_transformer_args(config: ModelArgs) -> TransformerArgs:
    return TransformerArgs(
        block_size=config.max_seq_len,
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dim=config.dim,
        n_local_heads=config.n_kv_heads if config.n_kv_heads is not None else config.n_heads,
        head_dim=config.dim // config.n_heads,
        rope_base=config.rope_theta,
        norm_eps=config.norm_eps,
        multiple_of=config.multiple_of,
        ffn_dim_multiplier=config.ffn_dim_multiplier,
        max_seq_length=config.max_seq_len,
        rope_scaling=None,  # Assuming no rope scaling for now
    )

def load_dcp_checkpoint(config: ModelArgs, checkpoint_path: str):
    job_config = JobConfig()
    job_config.parse_args([])

    environ["MASTER_ADDR"] = "localhost"
    environ["MASTER_PORT"] = "29500"
    environ["RDZV_BACKEND"] = "c10d"
    environ["WORLD_SIZE"] = str(1)
    environ["RANK"] = str(0)
    environ["LOCALRANK"] = str(0)

    if not dist.is_initialized():
        init_distributed()

    device_type, device_module = get_device_info()
    device = torch.device(device_type)

    set_determinism(None, device, seed=42)

    checkpoint_dir = Path(checkpoint_path)
    step_folder = checkpoint_dir.name
    dump_folder = checkpoint_dir.parent.parent

    job_config.checkpoint.enable_checkpoint = True
    job_config.checkpoint.folder = 'checkpoint'
    job_config.job.dump_folder = str(dump_folder)
    job_config.checkpoint.use_tensor_preload = False

    # Initialize the model using TorchTitan's approach
    model_config = llama3_configs["3B"]  # You can adjust the size as needed
    model_config.norm_type = "rmsnorm"
    model_config.vocab_size = config.vocab_size
    model_config.max_seq_len = config.max_seq_length or 8192

    # Convert ModelArgs to TransformerArgs
    transformer_args = convert_model_args_to_transformer_args(model_config)
    with torch.device("meta"):
        model = TorchchatTransformer(transformer_args)
        model = model.to(dtype=torch.bfloat16)

    # Initialize ParallelDims
    world_size = dist.get_world_size()
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        world_size=world_size,
        enable_loss_parallel=False
    )

    # Build device mesh
    world_mesh = parallel_dims.build_mesh(device_type)

    # Apply parallelism
    parallelize_llama(model, world_mesh, parallel_dims, job_config)

    # Move the model to the appropriate device
    model = model.to_empty(device=device)

    class ModelWrapper(Stateful):
        def __init__(self, model):
            self.model = model

        def state_dict(self):
            return self.model.state_dict()

        def load_state_dict(self, state_dict):
            self.model.load_state_dict(state_dict)

    class MinimalOptimizersContainer(Stateful):
        def __init__(self, model):
            self.optimizers = [Adam(model.parameters())]

        def state_dict(self):
            return {"optimizer_0": self.optimizers[0].state_dict()}

        def load_state_dict(self, state_dict):
            self.optimizers[0].load_state_dict(state_dict["optimizer_0"])

    class MinimalSchedulersContainer:
        def __init__(self, optimizer):
            self.schedulers = [LambdaLR(optimizer, lr_lambda=lambda _: 1)]

        def get_lr_scheduler_state(self):
            return {"lr_scheduler": self.schedulers[0]}

    model_wrapper = ModelWrapper(model)
    optimizers = MinimalOptimizersContainer(model)
    lr_schedulers = MinimalSchedulersContainer(optimizers.optimizers[0])

    train_state = TrainState()

    checkpoint_manager = CheckpointManager(
        dataloader=None,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state, "model": model_wrapper},
        job_config=job_config,
    )

    step = int(step_folder.split('-')[-1])
    success = checkpoint_manager.load(step=step)

    if not success:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    model.text_transformer_args = model_config
    model.text_transformer_args.max_seq_length = model_config.max_seq_len
    model.text_transformer_args = transformer_args

    torchchat_model_args = TorchchatModelArgs(
        model_type=ModelType.TextOnly,
        transformer_args={"text": transformer_args.__dict__},
        use_tiktoken=False,
        use_hf_tokenizer=False,
        tokenizer_prepend_bos=True,
    )
    m = TextOnlyModel(torchchat_model_args, skip_model_init=True)
    m.model = model
    m.text_transformer_args = transformer_args

    return m