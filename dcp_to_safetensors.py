#!/usr/bin/env python3
"""
DCP to safetensors converter using the same approach as dcp_util.py

Usage:
    python dcp_to_safetensors.py \
        --dcp-dir /path/to/checkpoint/step-0 \
        --output-dir /path/to/output \
        --tokenizer-path /path/to/tokenizer/dir \
        --model-size 3B
"""

import argparse
import json
import os
from pathlib import Path

import pydevd_pycharm
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from typing import Dict, Any
import torch.distributed as dist
import shutil

# Import the same modules as dcp_util.py
from torchchat.model import Transformer as TorchchatTransformer, TransformerArgs, ModelArgs as TorchchatModelArgs, \
    ModelType
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.models.llama import llama3_configs
from torchtitan.utils import get_device_info, set_determinism
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def parse_args():
    parser = argparse.ArgumentParser(description="DCP to safetensors converter")
    parser.add_argument("--dcp-dir", type=Path, required=True, help="Path to DCP checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--tokenizer-path", type=Path, required=True,
                        help="Path to directory containing tokenizer files (tokenizer.model, tokenizer.json, etc.)")
    parser.add_argument("--model-size", type=str, default="3B", help="Model size (e.g., 3B, 8B, 70B)")
    parser.add_argument("--max-shard-size", type=str, default="10GB", help="Max size per shard")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="Convert weights to this dtype (auto keeps original)")
    parser.add_argument("--vocab-size", type=int, default=128256, help="Vocabulary size")
    parser.add_argument("--download-hf-tokenizer", action="store_true",
                        help="Download tokenizer from HuggingFace instead of using local files")
    return parser.parse_args()


def parse_size_to_bytes(size_str: str) -> int:
    """Convert size string like '5GB' to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    else:
        return int(size_str)


def convert_model_args_to_transformer_args(config) -> TransformerArgs:
    """Convert ModelArgs to TransformerArgs (same as dcp_util.py)"""
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


def load_checkpoint_to_model(checkpoint_path: Path, model_size: str, vocab_size: int):
    """Load DCP checkpoint following the same pattern as dcp_util.py"""

    # Set up environment (non-distributed)
    os.environ.update({
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    })

    # Get device info
    device_type_str, _ = get_device_info()
    device = torch.device(device_type_str)
    set_determinism(None, device, seed=42)

    # Setup JobConfig
    job_config = JobConfig()
    job_config.parse_args([])

    # Setup checkpoint paths
    checkpoint_dir = Path(checkpoint_path)
    step_folder = checkpoint_dir.name
    dump_folder = checkpoint_dir.parent.parent

    job_config.checkpoint.enable_checkpoint = True
    job_config.checkpoint.folder = "checkpoint"
    job_config.job.dump_folder = str(dump_folder)
    job_config.checkpoint.use_tensor_preload = False

    # Get model config
    if model_size not in llama3_configs:
        raise ValueError(f"Model size '{model_size}' not found in llama3_configs")

    model_config = llama3_configs[model_size]
    model_config.norm_type = "rmsnorm"
    model_config.vocab_size = vocab_size
    model_config.max_seq_len = 8192  # Default

    # Convert to TransformerArgs
    transformer_args = convert_model_args_to_transformer_args(model_config)

    # Create model on meta device (same as dcp_util.py)
    with torch.device("meta"):
        model = TorchchatTransformer(transformer_args)
        model = model.to(dtype=torch.bfloat16)

    # Move to actual device
    model = model.to_empty(device=device)

    # Create the same wrapper classes as dcp_util.py
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

    # Create wrappers
    model_wrapper = ModelWrapper(model)
    optimizers = MinimalOptimizersContainer(model)
    lr_schedulers = MinimalSchedulersContainer(optimizers.optimizers[0])
    train_state = TrainState()

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        dataloader=None,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state, "model": model_wrapper},
        job_config=job_config,
    )

    # Load checkpoint
    step = int(step_folder.split('-')[-1])
    success = checkpoint_manager.load(step=step)

    if not success:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    return model, model_config


def convert_dtype(state_dict, target_dtype_str):
    """Convert state dict to target dtype."""
    if target_dtype_str == "auto":
        return state_dict

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map[target_dtype_str]

    print(f"Converting weights to {target_dtype_str}")
    converted = {}
    for key, tensor in tqdm(state_dict.items(), desc="Converting dtype"):
        if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            converted[key] = tensor.to(target_dtype)
        else:
            converted[key] = tensor

    return converted


def create_tokenizer_config_files(output_dir: Path):
    """Create the necessary tokenizer configuration files for vLLM."""

    # 1. Create tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "added_tokens_decoder": {
            "128000": {"content": "<|begin_of_text|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128001": {"content": "<|end_of_text|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128002": {"content": "<|reserved_special_token_0|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128003": {"content": "<|reserved_special_token_1|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128004": {"content": "<|reserved_special_token_2|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128005": {"content": "<|reserved_special_token_3|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128006": {"content": "<|start_header_id|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128007": {"content": "<|end_header_id|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128008": {"content": "<|reserved_special_token_4|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True},
            "128009": {"content": "<|eot_id|>", "lstrip": False, "normalized": False, "rstrip": False,
                       "single_word": False, "special": True}
        },
        "bos_token": "<|begin_of_text|>",
        "clean_up_tokenization_spaces": True,
        "eos_token": "<|eot_id|>",
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": 131072,
        "pad_token": "<|reserved_special_token_0|>",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
    }

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print("  Created: tokenizer_config.json")

    # 2. Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": {
            "content": "<|begin_of_text|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "eos_token": {
            "content": "<|eot_id|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "pad_token": {
            "content": "<|reserved_special_token_0|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        }
    }

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    print("  Created: special_tokens_map.json")

    # 3. Create generation_config.json
    generation_config = {
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128009],
        "pad_token_id": 128002,
        "do_sample": True,
        "temperature": 0.6,
        "max_length": 8192,
        "top_p": 0.9
    }

    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    print("  Created: generation_config.json")


def download_hf_tokenizer(output_dir: Path, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Download tokenizer from HuggingFace."""
    try:
        from transformers import AutoTokenizer
        print(f"\nDownloading tokenizer from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"  Successfully downloaded tokenizer to: {output_dir}")
        return True
    except Exception as e:
        print(f"  Failed to download tokenizer: {e}")
        print("  You may need to authenticate with HuggingFace or use a different model name")
        return False


def copy_tokenizer_files(tokenizer_path: Path, output_dir: Path):
    """Copy tokenizer files from specified tokenizer directory to output."""
    # Common tokenizer file patterns
    tokenizer_patterns = [
        "tokenizer.model",  # SentencePiece model
        "tokenizer.json",  # HF tokenizer
        "tokenizer_config.json",  # HF tokenizer config
        "special_tokens_map.json",  # Special tokens mapping
        "added_tokens.json",  # Additional tokens
        "vocab.json",  # Vocabulary
        "merges.txt",  # BPE merges
        "*.spm",  # Any SentencePiece model
        "*.tiktoken",  # Tiktoken files
    ]

    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")

    if not tokenizer_path.is_dir():
        raise ValueError(f"Tokenizer path must be a directory: {tokenizer_path}")

    print(f"\nCopying tokenizer files from: {tokenizer_path}")
    copied_files = []

    # Copy exact matches and glob patterns
    for pattern in tokenizer_patterns:
        if '*' in pattern:
            # Handle glob patterns
            for file in tokenizer_path.glob(pattern):
                if file.is_file():
                    shutil.copy2(file, output_dir / file.name)
                    copied_files.append(file.name)
                    print(f"  Copied: {file.name}")
        else:
            # Handle exact file names
            src = tokenizer_path / pattern
            if src.exists() and src.is_file():
                shutil.copy2(src, output_dir / pattern)
                copied_files.append(pattern)
                print(f"  Copied: {pattern}")

    if not copied_files:
        print(f"  WARNING: No tokenizer files found in {tokenizer_path}")
        print(f"  Looked for: {', '.join(tokenizer_patterns)}")
    else:
        print(f"  Total files copied: {len(copied_files)}")

    return copied_files


def calculate_intermediate_size(dim, ffn_dim_multiplier, multiple_of):
    """Calculate the intermediate size for FFN layers following LLaMA's convention."""
    # This follows the exact calculation from the LLaMA model
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def save_model_config(output_dir: Path, model_size: str, vocab_size: int, model_config):
    """Save config.json in the format of config_model.json for vLLM compatibility."""
    # Calculate the actual intermediate size using the same logic as the model
    intermediate_size = calculate_intermediate_size(
        model_config.dim,
        model_config.ffn_dim_multiplier,
        model_config.multiple_of
    )

    # Determine max_position_embeddings based on rope_scaling
    # For llama3 models, if rope_scaling is used, max_position_embeddings is typically larger
    max_position_embeddings = 131072 if model_size in ["3B", "8B", "70B", "405B"] else model_config.max_seq_len

    # Create config.json in the extended Hugging Face format (like config_model.json)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128009],  # Both end_of_text and eot_id
        "pad_token_id": 128004,  # Common Llama 3 pad token
        "hidden_act": "silu",
        "hidden_size": model_config.dim,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": model_config.n_heads,
        "num_hidden_layers": model_config.n_layers,
        "num_key_value_heads": model_config.n_kv_heads if model_config.n_kv_heads is not None else model_config.n_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": model_config.norm_eps,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        } if max_position_embeddings > 8192 else None,
        "rope_theta": float(model_config.rope_theta),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.42.3",
        "use_cache": True,
        "vocab_size": vocab_size
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved config.json in extended format for vLLM compatibility")
    print(f"  Hidden size: {model_config.dim}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Max position embeddings: {max_position_embeddings}")
    print(f"  Rope scaling: {'enabled' if config['rope_scaling'] else 'disabled'}")


def inspect_model_weights(state_dict):
    """Inspect model weights to understand the structure."""
    print("\n=== Model Weight Inspection ===")

    # Look for FFN weights
    ffn_weights = {}
    for key, tensor in state_dict.items():
        if 'feed_forward' in key or 'ffn' in key or 'w1' in key or 'w2' in key or 'w3' in key:
            ffn_weights[key] = tensor.shape

    if ffn_weights:
        print("\nFFN layer weights found:")
        for key, shape in sorted(ffn_weights.items())[:10]:  # Show first 10
            print(f"  {key}: {shape}")

    # Print dimensions of first FFN layer
    for i in range(100):  # Check first 100 layers
        w1_key = f"layers.{i}.feed_forward.w1.weight"
        w2_key = f"layers.{i}.feed_forward.w2.weight"
        w3_key = f"layers.{i}.feed_forward.w3.weight"

        if w1_key in state_dict:
            print(f"\nLayer {i} FFN dimensions:")
            print(f"  w1 (gate): {state_dict[w1_key].shape}")
            if w2_key in state_dict:
                print(f"  w2 (down): {state_dict[w2_key].shape}")
            if w3_key in state_dict:
                print(f"  w3 (up): {state_dict[w3_key].shape}")
            break

    print("==============================\n")


def save_as_safetensors(state_dict, output_dir: Path, max_shard_size: int):
    """Save state dict as sharded safetensors files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out non-tensor entries and special keys
    tensor_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Skip freqs_cis buffer
            if key == "freqs_cis":
                print(f"Skipping {key} buffer")
                continue
            tensor_dict[key] = value
        else:
            print(f"Skipping non-tensor entry: {key} (type: {type(value)})")

    if not tensor_dict:
        raise ValueError("No tensors found in state dict!")

    # Group into shards
    shards = []
    current_shard = {}
    current_size = 0

    print("Organizing into shards...")
    for key, tensor in tensor_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_shard and current_size + tensor_size > max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Save shards
    print(f"Saving {len(shards)} shards...")
    weight_map = {}

    if len(shards) == 1:
        # Single file
        output_path = output_dir / "model.safetensors"
        save_file(shards[0], output_path)
        print(f"Saved single file: {output_path}")
    else:
        # Multiple shards
        for i, shard in enumerate(tqdm(shards, desc="Saving shards")):
            shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            shard_path = output_dir / shard_name
            save_file(shard, shard_path)

            for key in shard.keys():
                weight_map[key] = shard_name

        # Save index
        index = {
            "metadata": {
                "total_size": sum(t.numel() * t.element_size() for t in tensor_dict.values()),
            },
            "weight_map": weight_map,
        }

        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Save basic config
    config = {
        "num_parameters": sum(t.numel() for t in tensor_dict.values()),
        "num_tensors": len(tensor_dict),
        "dtype": str(next(iter(tensor_dict.values())).dtype),
    }

    with open(output_dir / "conversion_info.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()

    # Validate tokenizer path (unless we're downloading from HF)
    if not args.download_hf_tokenizer:
        if not args.tokenizer_path.exists():
            print(f"ERROR: Tokenizer path does not exist: {args.tokenizer_path}")
            return 1

        if not args.tokenizer_path.is_dir():
            print(f"ERROR: Tokenizer path must be a directory: {args.tokenizer_path}")
            return 1

    # Initialize distributed if not already initialized
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print(f"Loading DCP checkpoint from {args.dcp_dir}")
    print(f"Model size: {args.model_size}, Vocab size: {args.vocab_size}")
    if not args.download_hf_tokenizer:
        print(f"Tokenizer path: {args.tokenizer_path}")

    # Load the model with checkpoint
    model, model_config = load_checkpoint_to_model(args.dcp_dir, args.model_size, args.vocab_size)

    # Get the state dict
    state_dict = model.state_dict()

    print(f"Loaded model with {len(state_dict)} parameters")
    total_params = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Inspect model weights to understand structure
    inspect_model_weights(state_dict)

    # Convert dtype if requested
    state_dict = convert_dtype(state_dict, args.dtype)

    # Save as safetensors
    max_shard_size = parse_size_to_bytes(args.max_shard_size)
    save_as_safetensors(state_dict, args.output_dir, max_shard_size)

    # Save config.json for vLLM with the model_config
    save_model_config(args.output_dir, args.model_size, args.vocab_size, model_config)

    # Handle tokenizer files
    print("\nSetting up tokenizer files...")

    if args.download_hf_tokenizer:
        # Try to download from HuggingFace
        success = download_hf_tokenizer(args.output_dir)
        if not success:
            print("  Falling back to creating tokenizer config files manually...")
            create_tokenizer_config_files(args.output_dir)
    else:
        # Copy tokenizer files from specified path
        copied_files = copy_tokenizer_files(args.tokenizer_path, args.output_dir)

        # Always create the additional config files to ensure vLLM compatibility
        print("\nCreating additional tokenizer configuration files...")
        create_tokenizer_config_files(args.output_dir)

        if not copied_files:
            print("\nWARNING: No tokenizer files were copied from the source directory.")
            print("The configuration files have been created, but you may need to add the actual tokenizer model file.")

    print(f"\nConversion complete! Output saved to: {args.output_dir}")
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values() if isinstance(t, torch.Tensor))
    print(f"Total size: {total_size / 1e9:.2f} GB")

    print("\nCreated files:")
    for file in sorted(args.output_dir.glob("*")):
        if file.is_file():
            print(f"  - {file.name}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()