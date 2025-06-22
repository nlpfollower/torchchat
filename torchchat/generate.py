# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import base64
import contextlib
import itertools
import logging
import os
import textwrap
import time
from concurrent import futures
from functools import partial

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch._C import _SDPBackend as SDPBackend

from PIL import Image
from torch.distributed.pipelining.schedules import _PipelineSchedule, _sorted_batch_p2p
from torch.distributed.pipelining.stage import _PipelineStageBase

# torchtune model definition dependencies
from torchtune.data import Message, padded_collate_tiled_images_and_mask

from torchtune.generation import sample as tune_sample

from torchtune.models.llama3_2_vision._model_builders import llama3_2_vision_transform
from torchtune.training import set_default_dtype

from torchchat.cli.builder import (
    _initialize_model,
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
from torchchat.distributed.utils import (
    Color as color,
    run_in_dist_env,
)
from torchchat.model import Model, ModelType
from torchchat.utils.build_utils import device_sync, set_precision
from torchchat.utils.device_info import get_device_info

logger = logging.getLogger(__name__)


# NOTE: Logging disabled by default here due to conflicts with torch._dynamo
class NoOpLogger:
    def __no_op(self, *_, **__):
        pass

    def __getattr__(self, name):
        return self.__no_op


logger = (
    NoOpLogger() if os.getenv("LOG_LEVEL") is None
    else logging.getLogger(__name__)
)


## Chat Formatters #############################################################

class _ChatFormatter(ABC):
    # Messages can arrive as a standard dict with "role" and "content" as
    # strings, or where "content" is a list of objects with "text" fields.
    MESSAGE_TYPE = Dict[str, Union[str, List[Dict[str, str]]]]

    # A dialog is a sequence of messages
    DIALOG_TYPE = List[MESSAGE_TYPE]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_dialog_prompt(
            self,
            dialog: DIALOG_TYPE,
            add_generation_prompt: bool = True,
    ) -> List[int]:
        """Encode a sequence of messages into a sequence of token IDs, including
        the chat template

        Args:
            dialog (DIALOG_TYPE): The sequence of dialog messages to encode.
                This will be the additional messages on top of those that have
                already been processed.
            add_generation_prompt (bool): Whether to include a generation prompt
                at the end of the encoded sequence.

        Returns:
            List[int]: A list of token IDs representing the encoded prompt.
        """


class Llama3ChatFormatter(_ChatFormatter):
    """Format a chat prompt using special tokens to demarcate roles and messages.

    Refer to the LLaMA3 documentation for more details https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    """

    def _encode_header(self, role) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def _encode_message(self, message: _ChatFormatter.MESSAGE_TYPE) -> List[int]:
        tokens = self._encode_header(message["role"])
        if isinstance(message["content"], str):
            tokens.extend(
                self.tokenizer.encode(message["content"], bos=False, eos=False)
            )
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    tokens.extend(
                        self.tokenizer.encode(content["text"], bos=False, eos=False)
                    )

        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(
            self,
            dialog: _ChatFormatter.DIALOG_TYPE,
            add_generation_prompt: bool = True,
    ) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self._encode_message(message))
        # Add the start of an assistant message for the model to complete.
        if add_generation_prompt and dialog and dialog[-1]["role"] != "assistant":
            tokens.extend(self._encode_header("assistant"))  # Pass role directly as a string
        return tokens


class Llama2ChatFormatter(_ChatFormatter):
    """
    Chat formatting for Llama2
    CITE: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/
    """

    B_INST, E_INST = "[INST] ", " [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    @staticmethod
    def _get_content_str(message: _ChatFormatter.MESSAGE_TYPE) -> str:
        if isinstance(message["content"], list):
            return message["content"][0]["text"]
        return message["content"]

    def encode_dialog_prompt(
            self,
            dialog: _ChatFormatter.DIALOG_TYPE,
            add_generation_prompt: bool = True,  # UNUSED
    ) -> List[int]:
        new_turn = True
        tokens = []
        for message in dialog:
            if new_turn:
                tokens += self.tokenizer.encode(f"{self.tokenizer.bos}{self.B_INST}")
            content = self._get_content_str(message).strip()
            role = message["role"]
            if role == "system":
                tokens += self.tokenizer.encode(f"{self.B_SYS}{content}{self.E_SYS}")
                new_turn = False
            elif role == "user":
                tokens += self.tokenizer.encode(f"{content}{self.E_INST}")
                new_turn = False
            elif role == "assistant":
                tokens += self.tokenizer.encode(f" {content} {self.tokenizer.eos}\n")
                new_turn = True
            else:
                raise ValueError("Invalid role in dialog.")
        return tokens


class HFTokenizerChatFormatter(_ChatFormatter):
    """Chat formatter that uses the built-in formatting capabilities of an HF
    tokenizer instance
    """

    def encode_dialog_prompt(
            self,
            dialog: _ChatFormatter.DIALOG_TYPE,
            add_generation_prompt: bool = True,
    ) -> List[int]:
        rendered = self.tokenizer.apply_chat_template(
            dialog, add_generation_prompt=add_generation_prompt
        )
        logger.debug("Formatted chat prompt:\n%s", rendered)
        return self.tokenizer.encode(rendered)


## Generation ##################################################################

@dataclass
class GeneratorArgs:
    prompt: Optional[str] = (
        None  # When passed into the Generator, this will be used as the system prompt
    )
    encoded_prompt: Optional[torch.Tensor] = None
    image_prompts: Optional[Sequence[Union[str, PathLike, bytes]]] = (
        None  # string or Path to an image file or the raw base64 bytes of an image
    )
    chat_mode: bool = False
    gui_mode: bool = False
    num_samples: int = 1
    max_new_tokens: int = 200
    top_k: int = 200
    temperature: float = 0.0  # deterministic argmax if 0.0
    compile: bool = False
    compile_prefill: bool = False
    speculate_k: int = 5
    sequential_prefill: bool = False
    max_autotune: bool = False
    # (Misnomer) See Issue: https://github.com/pytorch/torchchat/issues/1273
    is_torchtune_model: bool = False

    def __post_init__(self):
        if self.compile_prefill and self.sequential_prefill:
            raise RuntimeError("prefill compilation requires parallel prefill")

    def validate_build(
            self, builder_args: BuilderArgs, model_description: str = "model"
    ):
        reason = ""
        model_type = ""
        if not self.sequential_prefill:
            reason = "parallel prefill"
        if self.compile_prefill:
            reason = "model compilation for prefill"
        if self.compile:
            reason = "model compilation"
        if builder_args.aoti_package_path:
            model_type = "PT2"
        if builder_args.dso_path:
            model_type = "DSO"
        if builder_args.pte_path:
            model_type = "PTE"
        if model_type and reason:
            raise RuntimeError(
                f"cannot perform {reason} because a {model_type} {model_description} is used"
            )

    @classmethod
    def from_args(cls, args):
        dso_path = getattr(args, "dso_path", None)
        pte_path = getattr(args, "pte_path", None)
        aoti_package_path = getattr(args, "aoti_package_path", None)
        sequential_prefill = (
                args.sequential_prefill or bool(aoti_package_path) or bool(pte_path) or bool(dso_path)
        )

        # Validate that all image prompts exist before expensive model load
        if image_prompts := getattr(args, "image_prompts", None):
            non_existent_image_prompts = [
                image_prompt
                for image_prompt in image_prompts
                if (not os.path.exists(image_prompt))
            ]
            if non_existent_image_prompts:
                raise RuntimeError(
                    f"Image prompt {non_existent_image_prompts} does not exist"
                )

        return cls(
            prompt=getattr(args, "prompt", ""),
            encoded_prompt=None,
            image_prompts=image_prompts,
            chat_mode=args.chat,
            gui_mode=args.gui,
            num_samples=getattr(args, "num_samples", 1),
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=args.compile,
            compile_prefill=args.compile_prefill,
            speculate_k=args.speculate_k,
            sequential_prefill=sequential_prefill,
            max_autotune=args.max_autotune,
            is_torchtune_model=args.model and args.model.endswith("tune"),
        )


class LocalGenerator:
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    Args:
        builder_args: Defines the model configuration
        speculative_builder_args: Defines the speculative model configuration for speculative decode
        tokenizer_args: Defines the tokenizer configuration for both the model and speculative model
        generator_args: Controls the generation parameters
        profile: A Path to a directory where the profiling results will be stored, if enabled.
        quantize: If True, quantize the model. Please refer to docs/quantization.md for details.
        draft_quantize: If True, quantize the draft model.
    """

    def __init__(
            self,
            builder_args: BuilderArgs,
            speculative_builder_args: BuilderArgs,
            tokenizer_args: TokenizerArgs,
            generator_args: GeneratorArgs,
            profile: Optional[Path],
            quantize: bool,
            draft_quantize: bool,
    ):
        torch._inductor.config.coordinate_descent_tuning = (
                builder_args.device != "cpu"
        )
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

        self.builder_args = builder_args
        self.speculative_builder_args = speculative_builder_args
        self.tokenizer_args = tokenizer_args
        self.profile = profile
        self.quantize = quantize
        self.draft_quantize = draft_quantize
        self.is_torchtune_model = generator_args.is_torchtune_model
        self.dtype = builder_args.precision
        self.get_user_input: Callable = input

        self.rank: Optional[int] = None

        print(
            f"Using device={self.builder_args.device} {get_device_info(self.builder_args.device)}"
        )
        set_precision(self.builder_args.precision)

        self.is_speculative = self.speculative_builder_args.checkpoint_path is not None

        if generator_args.chat_mode and not self.builder_args.is_chat_model:
            # fmt: off
            print(textwrap.dedent(
                """
                *******************************************************
                This model is not known to support the chat function
                and may produce nonsensical or false output.
                *******************************************************
                """
            ))
            # fmt: on
        self.system_prompt = generator_args.prompt
        self.tokenizer = _initialize_tokenizer(self.tokenizer_args)

        # Right now the assumption is only llama3 uses tiktokenizer and it
        # must use tiktokenizer.
        # Piggy backing off of this flag then for now to identify llama3
        # without prompting user.
        self.is_llama3_model = self.tokenizer_args.is_tiktoken
        if self.is_llama3_model:
            self.chat_formatter = Llama3ChatFormatter(self.tokenizer)
            if generator_args.chat_mode:
                logger.debug(
                    "Llama3 model detected in chat mode. Using updated sentence schemas"
                )
        elif self.tokenizer_args.is_hf_tokenizer:
            if not self.tokenizer.has_chat_template():
                raise ValueError("Tokenizer must have a chat template")
            self.chat_formatter = HFTokenizerChatFormatter(self.tokenizer)
        else:
            self.chat_formatter = Llama2ChatFormatter(self.tokenizer)

        self.builder_args.setup_caches = False
        self.model = _initialize_model(self.builder_args, self.quantize, self.tokenizer)

        if self.is_speculative:
            self.draft_model = _initialize_model(
                self.speculative_builder_args,
                (
                    self.quantize
                    if self.draft_quantize == "quantize"
                    else self.draft_quantize
                ),
                self.tokenizer,
            )
        else:
            self.draft_model = None

        # torchtune model does not contain essential info for validation
        # TODO: refactor model config to be more generic
        if not self.is_torchtune_model:
            self.tokenizer_args.validate_model(self.model)
        self.tokenizer_args.validate_model(self.draft_model, "draft model")
        generator_args.validate_build(self.builder_args)
        generator_args.validate_build(self.speculative_builder_args, "draft model")

    def multinomial_sample_one_no_sync(
            self,
            probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def logits_to_probs(
            self, logits, temperature: float = 1.0, top_k: Optional[int] = None
    ):
        logits = logits / max(
            temperature, 1e-5 if logits.dtype != torch.float16 else 1e-3
        )

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
            self,
            logits,
            need_probs: bool,
            temperature: float = 0,
            top_k: Optional[int] = None,
    ):
        logits = logits[0, -1]
        logger.debug("Logits: %s", logits)
        if temperature == 0 and not need_probs:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
            return (idx_next, None)
        probs = self.logits_to_probs(logits, temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def prefill(
            self,
            model: Model,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
            *,
            sequential_prefill=True,
            **sampling_kwargs,
    ) -> torch.Tensor:
        logger.debug("x: %s, input_pos: %s", x, input_pos)

        # Add this debug
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"[Rank {rank}] Prefill input: x.shape={x.shape}, x={x}", flush=True)
            print(f"[Rank {rank}] Prefill input_pos: shape={input_pos.shape}, values={input_pos}", flush=True)

            # Check if all ranks have the same input
            x_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(x_list, x)
            if rank == 0:
                for i, tensor in enumerate(x_list):
                    print(f"  Rank {i} has x.shape={tensor.shape}", flush=True)
        width = x.size(1)
        assert input_pos.size(0) == width

        if self.model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"

            # TODO: Verify sequential prefill works with multimodal models
            is_multimodal = True
            if "encoder_input" in batch:
                encoder_input = batch["encoder_input"]
                encoder_mask = batch["encoder_mask"]
                is_multimodal = True
            else:
                encoder_input = None
                encoder_mask = None
                is_multimodal = False

            seq_len = x.size(1)
            mask = batch["causal_mask"][None, :seq_len]
            input_pos = input_pos.view(1, -1)
            logits = model(
                tokens=x,
                mask=mask,
                encoder_input=encoder_input,
                input_pos=input_pos,
                encoder_mask=encoder_mask,
            )[:, -1]

            if is_multimodal:
                batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

            return tune_sample(logits, temperature=0, top_k=500)
        elif sequential_prefill:
            for i in range(width):
                x_sliced, ip_sliced = x[:, i].view(-1, 1), input_pos[i].view(-1)
                logger.debug("<sliced> x: %s, input_pos: %s", x_sliced, ip_sliced)
                logits = model(x_sliced, ip_sliced)  # (x[:, i], input_pos[i])da
        else:
            # input_pos: [B, S]
            logits = model(x, input_pos)

        return self.sample(logits, need_probs=False, **sampling_kwargs)[0]

    def decode_one_token(
            self,
            model: Model,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            need_probs: bool,
            batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
            **sampling_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        x = x.view(1, -1)
        if model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"
            mask = batch["causal_mask"][None, input_pos.item(), None, :]
            encoder_mask = batch["encoder_mask"] if "encoder_mask" in batch else None
            logits = model(
                x, encoder_mask=encoder_mask, mask=mask, input_pos=input_pos
            )[:, -1:]
        else:
            logits = model(x, input_pos)
        return self.sample(logits, need_probs=need_probs, **sampling_kwargs)

    """
    Decode the next n tokens.

    Yields a tuple of (token, prob) for each token.
    """

    def decode_n_tokens(
            self,
            model: Model,
            cur_token: torch.Tensor,
            input_pos: torch.Tensor,
            num_new_tokens: int,
            need_probs: bool,
            batch=Optional[Dict[str, Any]],
            callback=lambda _: _,
            eos_token_id: int = 2,
            eot_id: Optional[int] = None,
            **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        encountered_eos = False

        # Check if we're in distributed mode with pipeline parallelism
        is_distributed_pp = hasattr(self, 'pp_rank') and self.builder_args.pp > 1

        for i in range(num_new_tokens - 1):
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                out_token = cur_token.clone()

                next_token, next_prob = self.decode_one_token(
                    model,
                    out_token,
                    input_pos,
                    batch=batch,
                    need_probs=need_probs,
                    **sampling_kwargs,
                )

                input_pos += 1

                # Only rank 0 has the real token
                if is_distributed_pp:
                    if self.pp_rank == self.first_pp_rank:
                        # Rank 0 checks for EOS
                        new_tokens.append(next_token.clone())
                        callback(new_tokens[-1], done_generating=i == num_new_tokens - 2)

                        # Check for EOS
                        is_eos = (next_token.item() == eos_token_id or
                                  (eot_id is not None and next_token.item() == eot_id))

                        # Broadcast EOS decision to all ranks
                        eos_tensor = torch.tensor([1 if is_eos else 0], device=self.device)
                        dist.broadcast(eos_tensor, src=self.first_pp_rank_global_id, group=self.pp_group)
                        encountered_eos = bool(eos_tensor.item())
                    else:
                        # Other ranks receive EOS decision
                        eos_tensor = torch.zeros(1, device=self.device, dtype=torch.long)
                        dist.broadcast(eos_tensor, src=self.first_pp_rank_global_id, group=self.pp_group)
                        encountered_eos = bool(eos_tensor.item())
                else:
                    # Non-distributed path (original logic)
                    new_tokens.append(next_token.clone())
                    callback(new_tokens[-1], done_generating=i == num_new_tokens - 2)

                    # Check for EOS
                    encountered_eos = (next_token.item() == eos_token_id or
                                       (eot_id is not None and next_token.item() == eot_id))

                if need_probs or next_prob is None:
                    yield out_token, None
                else:
                    new_probs.append(next_prob.clone())
                    yield out_token, next_prob.clone()

                # Update cur_token for next iteration
                cur_token = next_token.view(-1)  # Ensure it's 1D

                # Handle EOS case
                if encountered_eos:
                    final_token, next_prob = self.decode_one_token(
                        model,
                        cur_token,
                        input_pos,
                        need_probs,
                        batch=batch,
                        **sampling_kwargs,
                    )
                    input_pos += 1
                    yield cur_token.clone(), next_prob.clone() if next_prob is not None else None
                    break

        # Handle case where we didn't encounter EOS
        if not encountered_eos:
            eos_token = torch.tensor(
                [eos_token_id if eot_id is None else eot_id],
                dtype=cur_token.dtype,
                device=cur_token.device,
            )
            new_tokens.append(eos_token.clone())
            eos_token_out, next_prob = self.decode_one_token(
                model,
                eos_token.view(-1),
                input_pos,
                need_probs,
                batch=batch,
                **sampling_kwargs,
            )
            input_pos += 1
            yield eos_token.clone(), next_prob.clone() if next_prob is not None else None

    def model_forward(self, model, x, input_pos):
        return model(x, input_pos)

    def speculative_decode(
            self,
            model: Model,
            draft_model: Model,
            cur_token: torch.Tensor,
            input_pos: int,
            speculate_k: int,
            batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
            **sampling_kwargs,
    ) -> torch.Tensor:
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor(
            [input_pos], dtype=torch.int64, device=cur_token.device
        )
        draft_tokens, draft_probs = self.decode_n_tokens(
            draft_model,
            cur_token,
            orig_input_pos.clone(),
            speculate_k,
            batch=batch,
            need_probs=True,
            **sampling_kwargs,
        )

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = self.model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(
                input_pos, input_pos + speculate_k + 1, device=cur_token.device
            ),
        )
        target_probs = self.logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
        rejected_locations = (
                torch.rand_like(accept_draft_prob) > accept_draft_prob
        ).nonzero()

        if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = self.multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            self.model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = self.multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])

    @torch.no_grad()
    def generate(
            self,
            model: Model,
            prompt: torch.Tensor,
            max_new_tokens: int,
            *,
            chat_mode: bool,
            batch: Optional[
                Dict[str, Any]
            ] = None,  # List of Image prompt tensors for multimodal models
            start_pos: int = 0,
            skip_cache_setup: bool = False,
            draft_model: Model,
            speculate_k: Optional[int] = 8,
            sequential_prefill=True,
            callback=lambda x: x,
            max_seq_length: int,
            seed: Optional[int] = None,
            **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        if seed:
            torch.manual_seed(seed)

        is_speculative = draft_model is not None
        device, dtype = prompt.device, prompt.dtype

        if len(prompt.shape) > 1:
            prompt = prompt.squeeze(0)
        prompt_length = prompt.size(0)
        max_new_tokens = min(max_new_tokens, max_seq_length - start_pos - prompt_length)
        # set up caches only if first inference
        if start_pos == 0:
            if not skip_cache_setup:
                model = model.to(device=device)
                with torch.device(device):
                    if (
                            self.is_torchtune_model
                            or self.model.config.model_type == ModelType.Flamingo
                    ):
                        # 6404 is one-gpu affordable max_seq_length for single image input
                        model.setup_caches(
                            batch_size=1,
                            dtype=self.dtype,
                            encoder_max_seq_len=6404,
                            decoder_max_seq_len=max_seq_length,
                        )
                    else:
                        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
                    if is_speculative and draft_model is not model:
                        draft_model.setup_caches(
                            max_batch_size=1,
                            max_seq_length=max_seq_length,
                        )
            if model.config.model_type == ModelType.Flamingo:
                model.reset_caches()

        input_pos = torch.arange(
            start_pos, prompt_length + start_pos, device=device, dtype=torch.int
        )

        prefill_t0 = time.perf_counter()
        next_token = self.prefill(
            model,
            prompt.view(1, -1),
            input_pos,
            batch=batch,
            sequential_prefill=sequential_prefill,
            **sampling_kwargs,
        )
        if is_speculative:
            self.prefill(
                draft_model,
                prompt.view(1, -1),
                input_pos,
                sequential_prefill=sequential_prefill,
                **sampling_kwargs,
            )

        time_to_first_token = time.perf_counter() - prefill_t0
        yield None, {"time_to_first_token": time_to_first_token}
        # max_new_tokens <= 2 means we are effectively not calling decode_n_tokens().
        callback(next_token.clone().view(-1), done_generating=max_new_tokens <= 2)

        input_pos = torch.tensor(
            [start_pos + prompt_length], device=device, dtype=torch.int
        )
        accept_counts = [0] * (
                speculate_k + 1
        )  # creates array of [0, 0, 0, ...] that is speculate_k + 1 long

        if is_speculative:
            input_pos = (
                input_pos.item()
            )  # for speculative decoding easier to keep on host
            while input_pos < max_new_tokens - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    model,
                    draft_model,
                    cur_token,
                    input_pos,
                    speculate_k,
                    batch=batch,
                    **sampling_kwargs,
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(max_new_tokens - input_pos - 1, len(next_tokens))
                for token in next_tokens[:num_added, ]:
                    callback(token)
                    yield token, None
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            generated_tokens = []
            for generated_token, _ in self.decode_n_tokens(
                    model,
                    next_token,
                    input_pos,
                    max_new_tokens - 1,
                    batch=batch,
                    callback=callback,
                    need_probs=False,
                    eos_token_id=self.tokenizer.eos_id() if self.tokenizer else 2,
                    eot_id=(
                            self.tokenizer.special_tokens["<|eot_id|>"]
                            if self.is_llama3_model
                            else None
                    ),
                    **sampling_kwargs,
            ):
                generated_tokens.append(generated_token.view(-1))
                yield generated_token, None

        generate_stats = {
            "accept_counts": accept_counts,
        }
        yield None, generate_stats

    def encode_tokens(self, string, bos=True, device="cpu"):
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        logger.debug("Size after encode_tokens: %d", len(tokens))
        logger.debug("Token IDs: %s", tokens)
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def _callback(self, x, *, buffer, done_generating):
        period_id = self.tokenizer.encode(".")[0]
        buffer.append(
            self.tokenizer.decode([period_id] + x.tolist())[1:]
        )
        if x.item() == self.tokenizer.eos_id():
            done_generating = True
        if (
                self.is_llama3_model
                and x.item() == self.tokenizer.special_tokens["<|eot_id|>"]
        ):
            done_generating = True
            buffer = buffer[:-1]  # drop the eot_id from the output buffer

        # Change: Print each token immediately instead of buffering
        if len(buffer) > 0:
            print("".join(buffer), end="", flush=True)  # Always flush
            buffer.clear()

    def _gen_model_input(
            self,
            prompt: Union[str | List[Any]],
            image_prompts: Optional[List[str | Image.Image]] = None,
            max_new_tokens: Optional[int] = None,
            max_seq_len: Optional[int] = 2048,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Convert prompt and image prompts into consumable model input args.

        When prompt is a list, the anticipated format is OpenAI API Inspired:
            [ ..., {"role": message["role"], "content": message["content"]}, ...]

        Args:
            prompt (Union[str, List[Any]]): Prompt or list of dialog.
            image_prompts (Optional[List[str | Image.Image]]): List of image prompts. Used only with Llama 3.2 11B.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Used only with Llama 3.2 11B.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, Any]]]: Encoded prompt and batch config for multimodal models.
        """

        # Text-Only model
        if self.model.config.model_type != ModelType.Flamingo:
            # Single String prompt
            if isinstance(prompt, str):
                encoded = self.encode_tokens(
                    prompt, bos=self.model.config.tokenizer_prepend_bos, device=self.builder_args.device
                )
            # List of dialog
            else:
                tokens = self.chat_formatter.encode_dialog_prompt(prompt)
                encoded = torch.tensor(
                    tokens, dtype=torch.int, device=self.builder_args.device
                )

            logger.debug(encoded)
            return encoded, None

        # Llama 3.2 11B
        assert (
                image_prompts is None or len(image_prompts) == 1
        ), "At most one image is supported at the moment"

        if image_prompts and isinstance(image_prompts[0], str):
            images = [Image.open(image_prompts[0])]
        else:
            images = None

        assert (
                max_new_tokens is not None
        ), "max_new_tokens must be specified for Flamingo models"

        # Wrap string prompts into a list
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        image_found = False
        messages = []
        for message in prompt:
            if isinstance(message["content"], str):
                if not image_found and image_prompts:
                    messages.append(
                        Message(
                            role=message["role"],
                            content=[
                                {"type": "image", "content": images[0]},
                                {"type": "text", "content": message["content"]},
                            ],
                        )
                    )
                    image_found = True
                else:
                    messages.append(Message(**message))

            elif isinstance(message["content"], list):
                images = None
                for content_dict in message["content"]:
                    if content_dict["type"] == "text":
                        prompt_arg = content_dict["text"]
                    elif content_dict["type"] == "image_url":
                        assert (
                                images is None
                        ), "At most one image is supported at the moment"

                        base64_decoded = base64.b64decode(
                            content_dict["image_url"].split(";base64,")[1]
                        )
                        images = [Image.open(BytesIO(base64_decoded))]
                        image_found = True

                is_multimodal = images is not None
                content = [{"type": "text", "content": prompt_arg}]

                if is_multimodal:
                    content = [{"type": "image", "content": images[0]}] + content

                messages.append(
                    Message(
                        role=message["role"],
                        content=content,
                    )
                )

        messages.append(
            Message(
                role="assistant",
                content="",
            )
        )

        transform = llama3_2_vision_transform(str(self.tokenizer_args.tokenizer_path))

        device = torch.device(device=self.builder_args.device)

        with device, set_default_dtype(self.dtype):
            data = transform({"messages": messages}, inference=True)

            if image_found:
                batch = padded_collate_tiled_images_and_mask(
                    [data], pad_direction="left", pad_max_images=1, pad_max_tiles=transform.max_num_tiles
                )
                encoded = batch.pop("tokens").to(device).view(-1)
                seq_len = encoded.size(0)
                batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
                batch["encoder_input"]["images"] = batch["encoder_input"]["images"].to(
                    self.dtype
                )

            else:
                encoded = torch.tensor(data["tokens"], device=device).view(-1)
                seq_len = encoded.size(0)
                batch = {}

            total_response_length = seq_len + max_new_tokens
            batch["causal_mask"] = torch.nn.functional.pad(
                torch.tril(
                    torch.ones(
                        size=(total_response_length, total_response_length),
                        dtype=torch.bool,
                    )
                ),
                (
                    0,
                    max_seq_len - total_response_length,
                    0,
                    max_seq_len - total_response_length,
                ),
                value=0,
            )

        logger.debug(encoded)
        return encoded, batch

    def chat(
            self,
            generator_args: GeneratorArgs,
    ):
        if generator_args.chat_mode:
            print("Starting Interactive Chat")

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        if self.builder_args.distributed:
            # During distributed inference the model gets sharded among the ranks
            # So we need to all reduce the model size to get the total model size
            model_size = torch.tensor(model_size, dtype=torch.int64, device=self.device)
            dist.all_reduce(model_size)
            model_size = model_size.item()

        if generator_args.compile:
            if self.builder_args.device == "cpu":
                if generator_args.max_autotune:
                    kwargs = {"mode": "max-autotune"}
                else:
                    kwargs = {}
            else:
                kwargs = {"mode": "reduce-overhead"}

            if self.is_speculative:
                self.model_forward = torch.compile(
                    self.model_forward, fullgraph=True, **kwargs
                )

            if self.model.config.model_type == ModelType.Flamingo:
                # Based on https://github.com/pytorch/torchtune/blob/57ab583c84c4a9dcacac23aeabc81f2a679670fe/torchtune/training/_compile.py#L42-L52
                from torchtune.modules import (
                    TransformerCrossAttentionLayer,
                    TransformerSelfAttentionLayer,
                )

                decoder = self.model.model.decoder
                for m in reversed(list(decoder.modules())):
                    if isinstance(m, TransformerSelfAttentionLayer) or isinstance(
                            m, TransformerCrossAttentionLayer
                    ):
                        m.compile()
            else:
                self.decode_one_token = torch.compile(
                    self.decode_one_token, fullgraph=True, **kwargs
                )

            if generator_args.compile_prefill:
                self.prefill = torch.compile(
                    self.prefill, fullgraph=True, dynamic=True, **kwargs
                )

        self.system_prompt = None
        # Set up our max_seq_length

        # This is a hack to get around the fact that different models have different ways to record their max_seq_length and might be wrong
        # TODO: unify the max_seq_length config representation.
        text_transformer_args = self.model.text_transformer_args
        max_seq_length = (
            text_transformer_args.max_seq_length if text_transformer_args else 2048
        )

        encoded, batch = self._gen_model_input(
            generator_args.prompt,
            generator_args.image_prompts,
            generator_args.max_new_tokens,
            max_seq_length,
        )

        if generator_args.chat_mode:
            print(
                f"Entering Chat Mode. Will continue chatting back and forth with the language model until the models max context length of {max_seq_length} tokens is hit or until the user says /bye"
            )
            get_system_prompt = self.get_user_input(
                "Do you want to enter a system prompt? Enter y for yes and anything else for no. \n"
            )
            if get_system_prompt == "y" or get_system_prompt == "Y":
                self.system_prompt = self.get_user_input("What is your system prompt? \n")

        # `is_torchtune_model` is a misnomer since it doesn't capture all
        # torchtune models (i.e. Flamingo)
        # See Issue: https://github.com/pytorch/torchchat/issues/1273
        elif (
                not generator_args.is_torchtune_model
                and self.model.config.model_type != ModelType.Flamingo
        ):
            max_seq_length = min(
                encoded.size(0) + generator_args.max_new_tokens,
                (
                    text_transformer_args.block_size
                    if text_transformer_args is not None
                    else 2048
                ),
                max_seq_length,
            )

        if self.draft_model is not None:
            max_seq_length += self.speculative_builder_args.speculate_k + 1

        aggregate_metrics = {
            "tokens_per_sec": [],
            "first_token_per_sec": [],
            "next_tokens_per_sec": [],
            "accept_counts": [],
        }
        start_pos = 0

        # arbitrarily large number as chat mode goes until max_seq length
        # or user exits
        num_samples = (
            generator_args.num_samples if not generator_args.chat_mode else 100000
        )
        for i in range(num_samples):
            device_sync(device=self.builder_args.device)
            is_first_sample: bool = i == 0
            if generator_args.chat_mode:
                prompt = self.get_user_input("User: ")
                if prompt == "/bye":
                    print("Exiting Chat.\n")
                    break

                # Encode the additional messages added in this dialog turn. If
                # this is the first turn, that includes any system prompt.
                messages_to_encode = []
                if is_first_sample and self.system_prompt:
                    messages_to_encode.append(
                        {"role": "system", "content": self.system_prompt}
                    )
                messages_to_encode.append({"role": "user", "content": prompt})
                encoded = self.chat_formatter.encode_dialog_prompt(
                    messages_to_encode, add_generation_prompt=True,
                )
                encoded = torch.tensor(
                    encoded, dtype=torch.int, device=self.builder_args.device
                )
                if encoded.size(0) + start_pos > max_seq_length:
                    print(
                        "This prompt would take us past the max_seq_length. Ending Conversation."
                    )
                    break

                print("Model: ", end="")

                buffer = []

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            else:
                assert not generator_args.chat_mode

                buffer = [generator_args.prompt]

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            if self.profile:
                from torch._inductor import config as inductor_config

                torch._inductor.config.profiler_mark_wrapper_call = True
                torch._inductor.config.cpp.enable_kernel_profile = True
            if i != generator_args.num_samples - 1 or not self.profile:
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            t0 = time.perf_counter()
            num_tokens_generated = 0
            local_token_tensor = []
            with prof:
                generator_func = self.generate(
                    self.model,
                    encoded,
                    generator_args.max_new_tokens,
                    draft_model=self.draft_model,
                    speculate_k=generator_args.speculate_k,
                    chat_mode=generator_args.chat_mode,
                    batch=batch,
                    callback=callback,
                    temperature=generator_args.temperature,
                    top_k=generator_args.top_k,
                    sequential_prefill=generator_args.sequential_prefill,
                    start_pos=start_pos,
                    skip_cache_setup=not is_first_sample,
                    max_seq_length=max_seq_length,
                )
                if generator_args.chat_mode:
                    start_pos += encoded.size(0)
                for token_tensor, metrics in generator_func:
                    if token_tensor is not None:
                        if os.getenv('DEBUG_CACHE'):
                            print(f"Token tensor: {token_tensor}")
                            local_token_tensor.append(token_tensor.tolist()[0])
                        start_pos += token_tensor.size(0)
                        num_tokens_generated += token_tensor.size(0)
                    if metrics is not None:
                        aggregate_metrics.update(metrics)
                    yield token_tensor, metrics
            jit_compile = is_first_sample and (
                    generator_args.compile or generator_args.compile_prefill
            )
            if os.getenv('DEBUG_CACHE'):
                print(f"local_token_tensor: {local_token_tensor}")
                print(self.tokenizer.decode(local_token_tensor))
            compilation_time = time.perf_counter() - t0
            device_sync(device=self.builder_args.device)
            t = time.perf_counter() - t0
            if hasattr(prof, "export_chrome_trace"):
                if self.builder_args.device == "cpu":
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                elif self.builder_args.device == "cuda":
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                else:
                    print(prof.key_averages().table(sort_by="self_xpu_time_total"))
                prof.export_chrome_trace(f"{self.profile}.json")

            if start_pos >= max_seq_length:
                print(
                    f"[Max Sequence Length {max_seq_length} Reached. Ending Conversation.]"
                )
                print("---------------------------------------------------")

            tokens_sec = (num_tokens_generated + 1) / t
            first_token_sec = 1 / aggregate_metrics.get("time_to_first_token", 0)
            next_tokens_sec = num_tokens_generated / (
                    t - aggregate_metrics.get("time_to_first_token", 0)
            )

            if jit_compile:
                print(
                    f"just-in-time compilation time (incl run time): {compilation_time:.2} seconds"
                )
            else:
                # aggregate_metrics will not append when is jit_compile, which will affect the average numbers.
                aggregate_metrics["tokens_per_sec"].append(tokens_sec)
                aggregate_metrics["first_token_per_sec"].append(first_token_sec)
                aggregate_metrics["next_tokens_per_sec"].append(next_tokens_sec)

            logging.info(
                f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\
                \nGenerated {num_tokens_generated} tokens \
                \nTime for inference {i + 1}: {t:.04f} sec total \
                \nTime to first token: {aggregate_metrics.get('time_to_first_token', 0):.04f} sec \
with {'sequential' if generator_args.sequential_prefill else 'parallel'} prefill.\
                \n\n      Total throughput: {tokens_sec:.04f} tokens/sec, {1 / tokens_sec:.04f} s/token \
                \nFirst token throughput: {first_token_sec:.04f} tokens/sec, {1 / first_token_sec:.04f} s/token \
                \n Next token throughput: {next_tokens_sec:.04f} tokens/sec, {1 / next_tokens_sec:.04f} s/token \
                    "
            )
            logging.info(
                f"\nBandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            if i == 0:
                logging.info(
                    f"*** This first iteration will include cold start effects for dynamic import, hardware caches{', JIT compilation' if jit_compile else ''}. ***"
                )
            print("\n========================================\n")
            if start_pos >= max_seq_length:
                if generator_args.chat_mode:
                    break

            if not generator_args.chat_mode:
                start_pos = 0

        if self.is_speculative:
            counts_aggregated = [
                sum(i) for i in zip(*aggregate_metrics["accept_counts"])
            ]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)]) / sum(counts_aggregated)}"
            )

        avg_tokens_sec = torch.mean(
            torch.tensor(aggregate_metrics["tokens_per_sec"])
        ).item()
        avg_first_token_sec = torch.mean(
            torch.tensor(aggregate_metrics["first_token_per_sec"])
        ).item()
        avg_next_tokens_sec = torch.mean(
            torch.tensor(aggregate_metrics["next_tokens_per_sec"])
        ).item()

        if not (
                torch.isnan(torch.tensor(avg_tokens_sec))
                or torch.isnan(torch.tensor(avg_first_token_sec))
                or torch.isnan(torch.tensor(avg_next_tokens_sec))
        ):
            print(
                f"\nWarning: Excluding compile in calculations \
                \n      Average tokens/sec (total): {avg_tokens_sec:.2f} \
                \nAverage tokens/sec (first token): {avg_first_token_sec:.2f} \
                \nAverage tokens/sec (next tokens): {avg_next_tokens_sec:.2f} \n\
                "
            )
        if torch.cuda.is_available():
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        if torch.xpu.is_available():
            print(f"Memory used: {torch.xpu.max_memory_reserved() / 1e9:.02f} GB")


class DistributedGenerator(LocalGenerator):
    def __init__(
            self,
            builder_args: BuilderArgs,
            speculative_builder_args: BuilderArgs,
            tokenizer_args: TokenizerArgs,
            generator_args: GeneratorArgs,
            profile: Optional[Path],
            quantize: bool,
            draft_quantize: bool,
    ):
        is_speculative = speculative_builder_args.checkpoint_path is not None
        assert is_speculative == False, "Distributed inference with pp > 1 does not support speculative inference yet."

        # Initialize distributed attributes first
        if dist.is_initialized():
            self.rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", self.rank % torch.cuda.device_count()))
            self.device = torch.device(f"cuda:{local_rank}")
            builder_args.device = f"cuda:{local_rank}"
            torch.cuda.set_device(self.device)
            print(f"DistributedGenerator: Rank {self.rank} using device: {self.device}")
        else:
            self.rank = 0
            self.device = torch.device(builder_args.device if builder_args.device else "cuda")

        # Call parent init
        super().__init__(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            profile,
            quantize,
            draft_quantize,
        )

        def distributed_input(prompt: str) -> str:
            if dist.get_rank() == 0:
                text = [input(prompt)]
            else:
                text = [None]
            dist.broadcast_object_list(text)
            return text[0]

        self.get_user_input: Callable = distributed_input

        if builder_args.pp > 1:
            # PP-specific initialization remains the same
            text_transformer_args = self.model.text_transformer_args
            max_seq_length = (
                text_transformer_args.max_seq_length if text_transformer_args else 2048
            )
            self.seqlen_prefill = max_seq_length

            logger.warn(f"{color.yellow}Pipeline parallelism is still experimental and might be slow{color.reset}")
            pp_mesh = self.model.device_mesh["pp"]

            self.pp_rank = pp_mesh.get_local_rank()
            self.pp_group = pp_mesh.get_group()
            self.pp_degree = pp_mesh.size()

            self.first_pp_rank = 0
            self.last_pp_rank = self.pp_degree - 1
            self.first_pp_rank_global_id = dist.get_global_rank(self.pp_group, self.first_pp_rank)
            self.last_pp_rank_global_id = dist.get_global_rank(self.pp_group, self.last_pp_rank)

            # Create unified pipeline stages
            self.prefiller = self.create_pipeline_stage("prefill")
            self.decoder = self.create_pipeline_stage("decode")

            # Add decode step counter for debugging
            self._decode_step_count = 0
            self._tokens_generated = []

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    # Helper function to get example inputs and outputs for the stages.
    def get_example_ins_outs(self, batch_size: int, seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function generates example inputs and outputs for the prefill and decode stages.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the example inputs and outputs.
        """
        model_dtype = torch.bfloat16
        mb_ids = torch.randint(
            0, self.model.config.vocab_size, (batch_size, seqlen), device=self.device
        )
        activation = torch.rand(
            batch_size, seqlen, self.model.config.dim, device=self.device, dtype=model_dtype
        )
        logits = torch.rand(
            batch_size, seqlen, self.model.config.vocab_size, device=self.device, dtype=model_dtype
        )
        example_inputs = (mb_ids if self.pp_rank == self.first_pp_rank else activation,)
        example_outputs = (logits if self.pp_rank == self.last_pp_rank else activation,)
        return example_inputs, example_outputs

    def create_pipeline_stage(self, stage_type="prefill"):
        """Creates a pipeline stage using our custom schedule."""
        num_microbatches = max(self.pp_degree, 2)

        # Determine sequence length based on stage type
        if stage_type == "prefill":
            seq_len = self.seqlen_prefill
            wrapped_model = PipelineWrapper(self.model, expected_seq_len=seq_len)
        else:  # decode
            seq_len = 1
            wrapped_model = PipelineWrapper(self.model)

        # Create example inputs/outputs for pipeline setup
        logger.debug(f"Creating pipeline stage for {stage_type} {self.pp_rank=}, {self.pp_degree=}")
        example_inputs, example_outputs = self.get_example_ins_outs(num_microbatches, seq_len)

        stage = PipelineStage(
            wrapped_model,
            self.pp_rank,
            self.pp_degree,
            self.device,
            input_args=example_inputs,
            output_args=example_outputs,
            group=self.pp_group,
        )

        # Use our custom schedule
        scheduler = SingleBatchPipelineSchedule(stage, num_microbatches)
        return scheduler

    def prefill(self, model, x, input_pos, batch=None, *, sequential_prefill=True, **sampling_kwargs):
        """Simplified prefill that processes only real data."""
        if self.builder_args.pp == 1:
            return super().prefill(model, x, input_pos, batch, sequential_prefill=sequential_prefill, **sampling_kwargs)

        pad_token_id = 128004  # TODO: Get from tokenizer
        prompt_length = x.size(1)
        num_microbatches = max(self.pp_degree, 2)

        # Create batch for pipeline scheduling (only first has real data)
        padded_seq = torch.full(
            (num_microbatches, self.seqlen_prefill),
            pad_token_id,
            dtype=torch.int64,
            device=self.device
        )
        padded_seq[0, :prompt_length] = x[0]

        # Create input positions
        input_pos_full = torch.arange(self.seqlen_prefill, device=self.device, dtype=torch.int)
        input_pos_batched = input_pos_full.unsqueeze(0).expand(num_microbatches, -1)

        # Pipeline kwargs
        kwargs = {
            "input_pos": input_pos_batched,
            "cache_lane": 0,  # Always use lane 0 for real data
            "actual_seq_len": prompt_length,
        }

        # Run pipeline
        if self.pp_rank == self.first_pp_rank:
            logits = self.prefiller.step(padded_seq, **kwargs)
        elif self.pp_rank == self.last_pp_rank:
            logits = self.prefiller.step(**kwargs)
        else:
            self.prefiller.step(**kwargs)

        # Sample from real data only
        if self.pp_rank == self.last_pp_rank:
            # Extract only the logit at the last real token position from the first sample
            relevant_logits = logits[0:1, prompt_length - 1:prompt_length, :]
            new_token = self.sample(relevant_logits, need_probs=False, **sampling_kwargs)[0]

            if self.pp_rank != self.first_pp_rank:
                dist.send(new_token, dst=self.first_pp_rank_global_id, group=self.pp_group)
        else:
            new_token = torch.zeros(1, 1, device=self.device, dtype=torch.int64)
            if self.pp_rank == self.first_pp_rank:
                dist.recv(new_token, src=self.last_pp_rank_global_id, group=self.pp_group)

        return new_token

    def decode_one_token(self, model, x, input_pos, need_probs, batch=None, **sampling_kwargs):
        """Simplified decode that processes only real data."""
        if self.builder_args.pp == 1:
            return super().decode_one_token(model, x, input_pos, need_probs, batch=batch, **sampling_kwargs)

        assert input_pos.shape[-1] == 1
        new_token = x.view(1, -1)

        # Create batch for pipeline scheduling
        num_microbatches = max(self.pp_degree, 2)
        if new_token.size(0) < num_microbatches:
            # Replicate token for pipeline scheduling
            new_token = new_token.repeat(num_microbatches, 1)

        # Create input_pos for all microbatches
        if input_pos.dim() == 1:
            input_pos = input_pos.unsqueeze(0)
        if input_pos.size(0) < num_microbatches:
            input_pos = input_pos.expand(num_microbatches, -1)

        kwargs = {
            "input_pos": input_pos,
            "cache_lane": 0,  # Always use lane 0
        }

        # Run pipeline
        if self.pp_rank == self.first_pp_rank:
            self.decoder.step(new_token, **kwargs)
        elif self.pp_rank == self.last_pp_rank:
            logits = self.decoder.step(**kwargs)

            # Sample from real data only
            logits = logits[0:1]  # First sample only
            new_token, _ = self.sample(logits, need_probs=need_probs, **sampling_kwargs)

            if self.pp_rank != self.first_pp_rank:
                dist.send(new_token, dst=self.first_pp_rank_global_id, group=self.pp_group)
        else:
            self.decoder.step(**kwargs)

        # Handle token reception
        if self.pp_rank == self.first_pp_rank:
            if self.pp_rank != self.last_pp_rank:
                new_token = torch.zeros(1, 1, device=self.device, dtype=torch.int64)
                dist.recv(new_token, src=self.last_pp_rank_global_id, group=self.pp_group)
                new_token = new_token[0]  # Convert to 1D
        else:
            new_token = torch.zeros(1, device=self.device, dtype=torch.int64)

        return new_token, None

    def sample(
            self,
            logits,
            need_probs: bool,
            temperature: float = 0,
            top_k: Optional[int] = None,
    ):
        if temperature == 0 and not need_probs:
            _, idx_next = torch.topk(logits[0, -1], k=1, dim=-1)
            return (idx_next, None)
        probs = self.logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)

        return idx_next, probs


class SingleBatchPipelineSchedule(_PipelineSchedule):
    """
    Custom pipeline schedule that internally processes only a single batch
    but maintains compatibility with pipeline parallelism infrastructure.

    This schedule:
    1. Accepts multiple microbatches as input (for compatibility)
    2. Only processes the first microbatch through the model
    3. Replicates the output to match expected microbatch count
    """

    def __init__(
            self,
            stage: _PipelineStageBase,
            n_microbatches: int,
            loss_fn: Optional[Callable] = None,
            output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
    ):
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_initialized = False
        self._stage.has_backward = self._has_backward

        # Override the stage's forward method to handle single batch
        self._original_forward = stage.submod.forward
        stage.submod.forward = self._wrapped_forward

        if n_microbatches < self._num_stages:
            raise ValueError(
                f"Number of microbatches ({n_microbatches}) must be >= number of stages ({self._num_stages})"
            )

    def _wrapped_forward(self, *args, **kwargs):
        """
        Wrapped forward that processes only the first sample and replicates output.
        """
        # Get the expected batch size from input
        if args and hasattr(args[0], 'shape'):
            expected_batch_size = args[0].shape[0]
        else:
            expected_batch_size = self._n_microbatches

        # Extract only the first sample
        real_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() > 0:
                real_args.append(arg[0:1])
            else:
                real_args.append(arg)

        # Process only real data
        output = self._original_forward(*real_args, **kwargs)

        # Replicate output to match expected batch size
        if isinstance(output, torch.Tensor) and output.shape[0] == 1 and expected_batch_size > 1:
            output = output.repeat(expected_batch_size, *([1] * (output.dim() - 1)))

        return output

    def _initialize_stage(self, args, kwargs):
        """Initialize stage with infrastructure for multiple microbatches."""
        self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs)
        if self._has_backward:
            self._stage._prepare_backward_infra(self._n_microbatches)
        self._stage_initialized = True

    def _check_inputs(self, arg_mbs, kwarg_mbs, target_mbs, losses):
        """Override parent's check to handle our single-batch processing."""
        # We'll create dummy microbatches internally
        if arg_mbs is None:
            arg_mbs = [args if hasattr(self, '_cached_args') else ()] * self._n_microbatches
        if kwarg_mbs is None:
            kwarg_mbs = [kwargs if hasattr(self, '_cached_kwargs') else {}] * self._n_microbatches
        return arg_mbs, kwarg_mbs

    def _step_microbatches(
            self,
            arg_mbs: Optional[list] = None,
            kwarg_mbs: Optional[list] = None,
            target_mbs: Optional[list] = None,
            losses: Optional[list] = None,
    ):
        """
        Implementation of abstract method.
        Processes microbatches but only computes on the first one.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Process all microbatches through the pipeline infrastructure
        fwd_sends_to_wait: List[dist.Work] = []

        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                # Receive activations
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                # Forward - our wrapped forward handles single batch processing
                output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])

                # Send activations
                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug(f"[Stage {self._stage.stage_index}] Forwarded microbatch {i}")

            # Compute loss if needed
            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for sends to complete
        for work in fwd_sends_to_wait:
            work.wait()

        # No backward for now (inference only)
        if self._has_backward:
            # Would implement backward here if needed
            pass

        # Update losses if needed
        self._update_losses(self._stage, losses)

    def step(self, *args, target=None, losses: Optional[list] = None, **kwargs):
        """
        Run pipeline schedule, processing only first microbatch internally.
        """
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Cache args for use in _check_inputs
        self._cached_args = args
        self._cached_kwargs = kwargs

        # For the stage, we need to split inputs properly
        if args or kwargs:
            # The stage expects microbatched inputs
            # But we'll only process the first one through our wrapped forward
            args_split = [args] * self._n_microbatches
            kwargs_split = [kwargs] * self._n_microbatches
        else:
            args_split = [()] * self._n_microbatches
            kwargs_split = [{}] * self._n_microbatches

        # Split targets if provided
        targets_split = None
        if target is not None:
            if hasattr(target, 'shape') and target.shape[0] >= self._n_microbatches:
                targets_split = list(torch.tensor_split(target, self._n_microbatches))
            else:
                targets_split = [target] * self._n_microbatches

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return output from last stage
        if self._stage.is_last and len(self._stage.output_chunks) > 0:
            # Return the first output chunk (real data)
            return self._stage.output_chunks[0]
        else:
            return None


class SimplePipelineWrapper(torch.nn.Module):
    """
    Simplified wrapper that handles batch dimension adjustments for pipeline parallelism.
    """

    def __init__(self, model, expected_seq_len=None):
        super().__init__()
        self.model = model
        self.expected_seq_len = expected_seq_len
        self._forward_count = 0

    def forward(self, tokens, input_pos=None, **kwargs):
        # Expected batch size from pipeline
        expected_batch_size = tokens.shape[0] if tokens.dim() > 0 else 1

        # Process only first sample
        real_tokens = tokens[0:1] if tokens.dim() > 0 else tokens
        real_input_pos = input_pos[0] if input_pos is not None and input_pos.dim() == 2 else input_pos

        # Remove pipeline-specific kwargs
        model_kwargs = kwargs.copy()
        for key in ['microbatch_indices', 'actual_seq_len', 'real_mask']:
            model_kwargs.pop(key, None)

        # Handle actual sequence length for prefill
        actual_seq_len = kwargs.get('actual_seq_len', None)
        if actual_seq_len is not None and actual_seq_len < real_tokens.size(1):
            real_tokens = real_tokens[:, :actual_seq_len]
            if real_input_pos is not None and real_input_pos.dim() == 1:
                real_input_pos = real_input_pos[:actual_seq_len]

        # Forward through model
        output = self.model(real_tokens, input_pos=real_input_pos, **model_kwargs)

        # Pad sequence length if needed (for prefill)
        if self.expected_seq_len is not None and output.size(1) < self.expected_seq_len:
            pad_length = self.expected_seq_len - output.size(1)
            padding = torch.zeros(
                output.size(0), pad_length, output.size(2),
                dtype=output.dtype, device=output.device
            )
            output = torch.cat([output, padding], dim=1)

        # Replicate to match expected batch size
        if expected_batch_size > 1 and output.size(0) == 1:
            output = output.repeat(expected_batch_size, 1, 1)

        self._forward_count += 1
        return output


class PipelineWrapper(nn.Module):
    """
    Unified wrapper that handles pipeline parallelism efficiently.
    Processes only real data and replicates outputs as needed.
    """

    def __init__(self, model, expected_seq_len=None):
        super().__init__()
        self.model = model
        self.expected_seq_len = expected_seq_len
        self._forward_count = 0

    def forward(self, tokens, input_pos=None, **kwargs):
        # Expected batch size from pipeline
        expected_batch_size = tokens.shape[0] if tokens.dim() > 0 else 1

        # Process only first sample
        real_tokens = tokens[0:1] if tokens.dim() > 0 else tokens
        real_input_pos = input_pos[0] if input_pos is not None and input_pos.dim() == 2 else input_pos

        # Remove pipeline-specific kwargs
        model_kwargs = kwargs.copy()
        for key in ['microbatch_indices', 'actual_seq_len', 'real_mask']:
            model_kwargs.pop(key, None)

        # Handle actual sequence length for prefill
        actual_seq_len = kwargs.get('actual_seq_len', None)
        if actual_seq_len is not None and actual_seq_len < real_tokens.size(1):
            real_tokens = real_tokens[:, :actual_seq_len]
            if real_input_pos is not None and real_input_pos.dim() == 1:
                real_input_pos = real_input_pos[:actual_seq_len]

        # Forward through model
        output = self.model(real_tokens, input_pos=real_input_pos, **model_kwargs)

        # Pad sequence length if needed (for prefill)
        if self.expected_seq_len is not None and output.size(1) < self.expected_seq_len:
            pad_length = self.expected_seq_len - output.size(1)
            padding = torch.zeros(
                output.size(0), pad_length, output.size(2),
                dtype=output.dtype, device=output.device
            )
            output = torch.cat([output, padding], dim=1)

        # Replicate to match expected batch size
        if expected_batch_size > 1 and output.size(0) == 1:
            output = output.repeat(expected_batch_size, 1, 1)

        self._forward_count += 1
        return output


def run_generator(
        args,
        rank: Optional[int] = None
):
    """
    This function creates and executes a generator
    """
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    # Setup rank 1 and up to suppress log messages and print messages
    if builder_args.distributed and rank != 0:
        logger.setLevel(logging.CRITICAL)
        context = contextlib.redirect_stdout(None)
    else:
        context = contextlib.nullcontext()

    with context:
        Generator = DistributedGenerator if builder_args.distributed else LocalGenerator
        logger.debug("GeneratorArgs: %s", generator_args)
        gen = Generator(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if torch.xpu.is_available():
            torch.xpu.reset_peak_memory_stats()

        for _ in gen.chat(generator_args):
            pass


def main(args):
    builder_args = BuilderArgs.from_args(args)

    if builder_args.distributed:
        # Check if we're already running under torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            run_generator(args, rank)
        else:
            # Calculate world size including dp
            world_size = builder_args.dp * builder_args.tp * builder_args.pp

            ctx = mp.get_context('spawn')
            with futures.ProcessPoolExecutor(max_workers=world_size - 1, mp_context=ctx) as executor:
                for i in range(1, world_size):
                    fn = partial(run_generator, args, i)
                    executor.submit(run_in_dist_env, world_size, i, fn)
                # Starting rank 0
                fn = partial(run_generator, args, 0)
                run_in_dist_env(world_size, 0, fn)
    else:
        run_generator(args)
