import os
from os import environ
import types
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.tensor import DTensor

from torchchat.model import TransformerArgs, ModelType
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.models.llama import TransformerModelArgs as ModelArgs, llama3_configs
from torchtitan.parallelisms import ParallelDims
from torchtitan.models.llama.parallelize_llama import parallelize_llama
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.llama.pipeline_llama import pipeline_llama_manual_split
from torchtitan.utils import device_type, get_device_info, set_determinism
from torchchat.model import Transformer as TorchchatTransformer, TextOnlyModel, ModelArgs as TorchchatModelArgs

# Import FSDP2 components
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
)


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
        rope_scaling=None,
    )


def load_dcp_checkpoint(
        config: TorchchatModelArgs,
        checkpoint_path: str,
        checkpoint_folder: str = "checkpoint",
        model_size: str = "3B",
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,  # Add dp parameter for FSDP
):
    """
    Load a distributed checkpoint with optional FSDP2 support.

    When dp > 1 and tp == 1, uses FSDP2 for data parallelism.
    When tp > 1, uses tensor parallelism (original behavior).
    """
    job_config = JobConfig()
    job_config.parse_args([])

    # Determine if we should use FSDP based on parameters
    use_fsdp = (dp > 1 and tp == 1)

    # Check if we're in distributed mode
    is_distributed = (dp > 1 or tp > 1 or pp > 1) or ("RANK" in environ and "WORLD_SIZE" in environ)

    # Initialize distributed if needed
    if is_distributed:
        if not dist.is_initialized():
            if "RANK" in environ and "WORLD_SIZE" in environ:
                dist.init_process_group(backend="nccl")
            else:
                environ["MASTER_ADDR"] = environ.get("MASTER_ADDR", "localhost")
                environ["MASTER_PORT"] = environ.get("MASTER_PORT", "29500")
                environ["RDZV_BACKEND"] = "c10d"
                # Calculate world size based on all parallelism dimensions
                world_size = dp * tp * pp if use_fsdp else tp * pp
                environ["WORLD_SIZE"] = str(world_size)
                environ["RANK"] = "0"
                environ["LOCAL_RANK"] = "0"
                dist.init_process_group(backend="nccl")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        environ["WORLD_SIZE"] = "1"
        environ["RANK"] = "0"
        environ["LOCAL_RANK"] = "0"

    device_type_str, device_module = get_device_info()

    # Set device
    if device_type_str == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_type_str)

    print(f"Rank {rank} using device: {device}")

    set_determinism(None, device, seed=42)

    # Setup checkpoint directory
    checkpoint_dir = Path(checkpoint_path)
    step_folder = checkpoint_dir.name
    dump_folder = checkpoint_dir.parent.parent

    job_config.checkpoint.enable_checkpoint = True
    job_config.checkpoint.folder = checkpoint_folder
    job_config.job.dump_folder = str(dump_folder)
    job_config.checkpoint.use_tensor_preload = False

    # Initialize model configuration
    if model_size not in llama3_configs:
        raise ValueError(f"Model size '{model_size}' not found in llama3_configs")

    model_config = llama3_configs[model_size]
    model_config.norm_type = "rmsnorm"
    model_config.vocab_size = config.vocab_size
    model_config.max_seq_len = config.max_seq_length or 8192

    # Convert ModelArgs to TransformerArgs
    transformer_args = convert_model_args_to_transformer_args(model_config)

    # Initialize ParallelDims based on whether we're using FSDP or TP
    if use_fsdp:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=dp,  # Use dp_shard for FSDP
            cp=1,
            tp=1,  # No tensor parallelism when using FSDP
            pp=pp,
            world_size=world_size,
            enable_loss_parallel=False
        )
    else:
        # Original TP behavior
        dp_degree = world_size // (tp * pp)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=dp_degree,
            cp=1,
            tp=tp,
            pp=pp,
            world_size=world_size,
            enable_loss_parallel=False
        )

    # Build device mesh
    if is_distributed:
        world_mesh = parallel_dims.build_mesh(device_type_str)
    else:
        world_mesh = None

    # Set stage-specific parameters for PP
    if pp > 1 and is_distributed:
        pp_mesh = world_mesh["pp"]
        pp_rank = pp_mesh.get_local_rank()
        transformer_args.stage_idx = pp_rank
        transformer_args.n_stages = pp
    else:
        transformer_args.stage_idx = 0
        transformer_args.n_stages = 1

    # Create the base model on meta device
    with torch.device("meta"):
        model = TorchchatTransformer(transformer_args)
        model = model.to(dtype=torch.bfloat16)

    # Apply pipeline split if PP is enabled
    if parallel_dims.pp_enabled and is_distributed:
        pp_mesh = world_mesh["pp"]
        stages, model_parts = pipeline_llama_manual_split(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
        )
        model = model_parts[0]
        print(f"Rank {rank} has layers: {list(model.layers.keys())}")
    else:
        model_parts = [model]

    # Apply parallelism based on configuration
    if is_distributed and (parallel_dims.tp_enabled or parallel_dims.dp_enabled):
        if use_fsdp:
            # Apply FSDP2 for data parallelism
            print(f"Applying FSDP2 with dp_shard={dp}")
            # Use the apply_fsdp function from torchtitan
            from torchtitan.models.llama.parallelize_llama import apply_fsdp

            dp_mesh = world_mesh["dp_shard"]
            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                pp_enabled=parallel_dims.pp_enabled,
                cpu_offload=False,
                reshard_after_forward_policy="always",  # For inference, always reshard
            )
        else:
            # Apply TP (original behavior)
            print(f"Applying Tensor Parallelism with tp={tp}")
            from torchtitan.models.llama.parallelize_llama import apply_tp

            apply_tp(
                model,
                world_mesh["tp"],
                loss_parallel=parallel_dims.loss_parallel_enabled,
                enable_float8=False,
                enable_async_tp=False,
            )

    # Move the model to the appropriate device
    model = model.to_empty(device=device)

    # Set text_transformer_args on the model
    model.text_transformer_args = transformer_args

    # Create wrappers for checkpoint loading
    class ModelWrapper(Stateful):
        def __init__(self, model):
            self.model = model

        def state_dict(self):
            return self.model.state_dict()

        def load_state_dict(self, state_dict):
            model_state = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}
            self.model.load_state_dict(filtered_state_dict, strict=False)

    class MinimalOptimizersContainer(Stateful):
        def __init__(self, model):
            self.optimizers = [Adam(model.parameters())]

        def state_dict(self):
            return {"optimizer_0": self.optimizers[0].state_dict()}

        def load_state_dict(self, state_dict):
            if "optimizer_0" in state_dict:
                try:
                    self.optimizers[0].load_state_dict(state_dict["optimizer_0"])
                except:
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

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        dataloader=None,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state, "model": model_wrapper},
        job_config=job_config,
    )

    step = int(step_folder.split('-')[-1])
    success = checkpoint_manager.load(step=step)

    if not success:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    # Create torchchat model args
    torchchat_model_args = TorchchatModelArgs(
        model_type=ModelType.TextOnly,
        transformer_args={"text": transformer_args.__dict__},
        use_tiktoken=False,
        use_hf_tokenizer=False,
        tokenizer_prepend_bos=True,
    )

    # Set attributes directly on the model
    model.device_mesh = world_mesh if is_distributed else None
    model.text_transformer_args = transformer_args
    model.config = transformer_args

    # Add PP-specific attributes if PP is enabled
    if pp > 1 and is_distributed:
        model.pp_rank = pp_rank
        model.pp_group = pp_mesh.get_group()
        model.pp_degree = pp
        model.first_pp_rank = 0
        model.last_pp_rank = pp - 1
        model.first_pp_rank_global_id = dist.get_global_rank(pp_mesh.get_group(), 0)
        model.last_pp_rank_global_id = dist.get_global_rank(pp_mesh.get_group(), pp - 1)

    # Create a simple namespace to hold model type info
    model.config.model_type = ModelType.TextOnly
    model.config.tokenizer_prepend_bos = torchchat_model_args.tokenizer_prepend_bos

    # For non-distributed mode, wrap in TextOnlyModel for compatibility
    if not is_distributed:
        m = TextOnlyModel(torchchat_model_args, skip_model_init=True)
        m.model = model
        m.text_transformer_args = transformer_args
        return m
    else:
        return model