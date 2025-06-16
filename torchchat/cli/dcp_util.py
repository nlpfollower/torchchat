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
from torchtitan.models.llama.pipeline_llama import pipeline_llama_manual_split
from torchtitan.utils import device_type, get_device_info, set_determinism
from torchchat.model import Transformer as TorchchatTransformer, TextOnlyModel, ModelArgs as TorchchatModelArgs


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


def load_dcp_checkpoint(config: ModelArgs, checkpoint_path: str, checkpoint_folder: str = "checkpoint",
                        model_size: str = "3B"):
    """
    Load a distributed checkpoint.

    Args:
        config: Model configuration
        checkpoint_path: Path to the checkpoint
        checkpoint_folder: Name of the checkpoint folder (default: "checkpoint")
        model_size: Model size to use from llama3_configs (default: "3B")
    """
    job_config = JobConfig()
    job_config.parse_args([])

    # Check if distributed is already initialized
    if not dist.is_initialized():
        # Check if we're running under torchrun (which sets these env vars)
        if "RANK" in environ and "WORLD_SIZE" in environ:
            # torchrun has already set up the environment, just init process group
            dist.init_process_group(backend="nccl")
        else:
            # We're not under torchrun, set up manually
            environ["MASTER_ADDR"] = environ.get("MASTER_ADDR", "localhost")
            environ["MASTER_PORT"] = environ.get("MASTER_PORT", "29500")
            environ["RDZV_BACKEND"] = "c10d"
            dist.init_process_group(backend="nccl")

    device_type, device_module = get_device_info()

    # Get the local rank for proper GPU assignment
    local_rank = int(environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
    device = torch.device(f"{device_type}:{local_rank}")

    # Set the CUDA device for this process
    if device_type == "cuda":
        torch.cuda.set_device(device)

    print(f"Rank {dist.get_rank()} using device: {device}")

    set_determinism(None, device, seed=42)

    checkpoint_dir = Path(checkpoint_path)
    step_folder = checkpoint_dir.name
    dump_folder = checkpoint_dir.parent.parent

    job_config.checkpoint.enable_checkpoint = True
    job_config.checkpoint.folder = checkpoint_folder
    job_config.job.dump_folder = str(dump_folder)
    job_config.checkpoint.use_tensor_preload = False
    print(f"checkpoint.folder: {job_config.checkpoint.folder}")
    print(f"dump folder: {job_config.job.dump_folder}")

    # Initialize the model using TorchTitan's approach
    if model_size not in llama3_configs:
        raise ValueError(
            f"Model size '{model_size}' not found in llama3_configs. Available sizes: {list(llama3_configs.keys())}")

    model_config = llama3_configs[model_size]
    model_config.norm_type = "rmsnorm"
    model_config.vocab_size = config.vocab_size
    model_config.max_seq_len = config.max_seq_length or 8192

    # Convert ModelArgs to TransformerArgs
    transformer_args = convert_model_args_to_transformer_args(model_config)

    # Initialize ParallelDims
    world_size = dist.get_world_size()
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=2,
        world_size=world_size,
        enable_loss_parallel=False
    )

    # Build device mesh
    world_mesh = parallel_dims.build_mesh(device_type)

    # Create the base model on meta device
    with torch.device("meta"):
        model = TorchchatTransformer(transformer_args)
        model = model.to(dtype=torch.bfloat16)

    # Apply pipeline split BEFORE loading checkpoint
    # This will create model_parts with only the layers for this rank
    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]
        stages, model_parts = pipeline_llama_manual_split(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
        )
        # Use the first model part for this rank
        model = model_parts[0]
        print(f"Rank {dist.get_rank()} has layers: {list(model.layers.keys())}")

    # Apply other parallelisms (TP, etc.)
    parallelize_llama(model, world_mesh, parallel_dims, job_config)

    # Move the model to the appropriate device
    model = model.to_empty(device=device)

    # Set text_transformer_args on the model
    model.text_transformer_args = transformer_args

    class ModelWrapper(Stateful):
        def __init__(self, model):
            self.model = model

        def state_dict(self):
            return self.model.state_dict()

        def load_state_dict(self, state_dict):
            # Only load the state dict entries that exist in this model
            model_state = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}
            self.model.load_state_dict(filtered_state_dict, strict=False)

    class MinimalOptimizersContainer(Stateful):
        def __init__(self, model):
            self.optimizers = [Adam(model.parameters())]

        def state_dict(self):
            return {"optimizer_0": self.optimizers[0].state_dict()}

        def load_state_dict(self, state_dict):
            # Only load optimizer states for parameters that exist
            if "optimizer_0" in state_dict:
                try:
                    self.optimizers[0].load_state_dict(state_dict["optimizer_0"])
                except:
                    # If optimizer state doesn't match, skip it (for seed checkpoints)
                    pass

    class MinimalSchedulersContainer:
        def __init__(self, optimizer):
            self.schedulers = [LambdaLR(optimizer, lr_lambda=lambda _: 1)]

        def get_lr_scheduler_state(self):
            return {"lr_scheduler": self.schedulers[0]}

    model_wrapper = ModelWrapper(model)
    optimizers = MinimalOptimizersContainer(model)
    lr_schedulers = MinimalSchedulersContainer(optimizers.optimizers[0])

    train_state = TrainState()

    # Create checkpoint manager with model_parts if PP is enabled
    checkpoint_manager = CheckpointManager(
        dataloader=None,
        model_parts=[model] if not parallel_dims.pp_enabled else model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state, "model": model_wrapper},
        job_config=job_config,
    )

    step = int(step_folder.split('-')[-1])
    success = checkpoint_manager.load(step=step)

    if not success:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    # Create torchchat model args (needed for both PP and non-PP cases)
    torchchat_model_args = TorchchatModelArgs(
        model_type=ModelType.TextOnly,
        transformer_args={"text": transformer_args.__dict__},
        use_tiktoken=False,
        use_hf_tokenizer=False,
        tokenizer_prepend_bos=True,
    )

    # For pipeline parallel, we need to return the raw transformer model
    # because the TextOnlyModel wrapper doesn't properly handle all the
    # pipeline-specific arguments like cache_lane
    if parallel_dims.pp_enabled:
        # Set attributes directly on the model
        model.device_mesh = world_mesh
        model.text_transformer_args = transformer_args
        model.config = transformer_args

        # Create a simple namespace to hold model type info
        model.config.model_type = ModelType.TextOnly
        model.config.tokenizer_prepend_bos = torchchat_model_args.tokenizer_prepend_bos

        return model
    else:
        # For non-PP cases, use the TextOnlyModel wrapper as before
        m = TextOnlyModel(torchchat_model_args, skip_model_init=True)
        m.model = model
        m.text_transformer_args = transformer_args

        # Set the required attributes for distributed inference
        m.device_mesh = world_mesh
        m.config = types.SimpleNamespace()
        m.config.model_type = ModelType.TextOnly
        m.config.vocab_size = model_config.vocab_size
        m.config.dim = model_config.dim
        m.config.n_heads = model_config.n_heads
        m.config.block_size = model_config.max_seq_len
        m.config.rope_base = model_config.rope_theta
        m.config.norm_eps = model_config.norm_eps
        m.config.tokenizer_prepend_bos = torchchat_model_args.tokenizer_prepend_bos

        return m