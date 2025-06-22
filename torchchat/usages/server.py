# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import atexit
import json
import logging
import os
import time
import threading
import queue

logger = logging.getLogger(__name__)

from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from flask import Flask, request, Response

from torchchat.cli.builder import BuilderArgs, TokenizerArgs
from torchchat.generate import GeneratorArgs

from torchchat.usages.openai_api import (
    CompletionRequest,
    get_model_info_list,
    create_openai_api_generator,
    retrieve_model_info,
    OpenAiApiGeneratorMixin,
)

OPENAI_API_VERSION = "v1"


# Extended mixin for distributed support
class DistributedOpenAiApiGeneratorMixin(OpenAiApiGeneratorMixin):
    """Extended mixin that handles distributed pipeline parallelism."""

    def chunked_completion(self, completion_request: CompletionRequest):
        """Distributed-aware chunked completion."""
        # Check if we're in distributed mode by looking for dist module
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            if rank != 0:
                # Non-zero ranks just participate in generation
                yield from self._distributed_generate(completion_request)
            else:
                # Rank 0 does normal processing
                yield from super().chunked_completion(completion_request)
        else:
            # Non-distributed path
            yield from super().chunked_completion(completion_request)

    def _distributed_generate(self, completion_request: CompletionRequest):
        """Non-zero ranks participate in generation without yielding."""
        # Encode inputs
        encoded, batch = self._gen_model_inputs_from_openai_completion_request(
            completion_request
        )

        # Set up generator args
        generator_args = GeneratorArgs(
            None,
            max_new_tokens=(
                int(completion_request.max_tokens)
                if completion_request.max_tokens
                else 16
            ),
            encoded_prompt=encoded,
            temperature=float(completion_request.temperature),
            chat_mode=False,
            sequential_prefill=True,
        )

        # Participate in generation but don't yield anything
        for _ in self.generate(
                model=self.model,
                prompt=encoded,
                max_new_tokens=generator_args.max_new_tokens,
                draft_model=self.draft_model,
                speculate_k=generator_args.speculate_k,
                chat_mode=generator_args.chat_mode,
                batch=batch,
                callback=lambda x, done_generating=False: None,
                temperature=generator_args.temperature,
                top_k=generator_args.top_k,
                sequential_prefill=generator_args.sequential_prefill,
                start_pos=0,
                max_seq_length=self.max_seq_length,
                seed=int(completion_request.seed or 0),
        ):
            pass  # Just participate, don't yield

        # Yield nothing for non-zero ranks
        return
        yield  # Make it a generator


def create_openai_api_generator_distributed(distributed: bool):
    """Create generator with distributed support."""
    if distributed:
        from torchchat.generate import DistributedGenerator
        return type('OpenAiApiGenerator', (DistributedOpenAiApiGeneratorMixin, DistributedGenerator), {})
    else:
        from torchchat.generate import LocalGenerator
        return type('OpenAiApiGenerator', (OpenAiApiGeneratorMixin, LocalGenerator), {})


def initialize_generator(args):
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    generator_args.chat_mode = False

    # Use our custom factory
    OpenAiApiGenerator = create_openai_api_generator_distributed(builder_args.distributed)

    return OpenAiApiGenerator(
        builder_args=builder_args,
        speculative_builder_args=speculative_builder_args,
        tokenizer_args=tokenizer_args,
        generator_args=generator_args,
        profile=args.profile,
        quantize=args.quantize,
        draft_quantize=args.draft_quantize,
    )


def worker_loop(gen):
    """Main loop for non-zero ranks."""
    rank = dist.get_rank()
    print(f"Rank {rank}: Entering worker loop...")

    while True:
        try:
            # Synchronize with rank 0
            dist.barrier()

            # Wait for broadcast request from rank 0
            req_list = [None]
            dist.broadcast_object_list(req_list, src=0)

            if req_list[0] == "SHUTDOWN":
                print(f"Rank {rank}: Received shutdown signal")
                break
            elif req_list[0] is not None:
                # Process the request
                req = req_list[0]
                print(f"Rank {rank}: Processing request for model {req.model}")

                if req.stream:
                    # For streaming, we need to consume the generator even though we don't yield
                    for _ in gen.chunked_completion(req):
                        pass
                else:
                    # For non-streaming, just call sync_completion
                    gen.sync_completion(req)

                print(f"Rank {rank}: Finished processing request")

        except KeyboardInterrupt:
            print(f"Rank {rank}: Interrupted")
            break
        except Exception as e:
            print(f"Rank {rank}: Error in worker loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


def create_app(args):
    """Creates a Flask app for the API server."""
    app = Flask(__name__)

    # Initialize generator
    gen = initialize_generator(args)
    builder_args = BuilderArgs.from_args(args)

    # Check if distributed
    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))

    # Non-zero ranks enter worker loop
    if is_distributed and rank != 0:
        worker_loop(gen)
        return None  # Never reached

    # Only rank 0 continues to set up Flask

    def _del_none(d: Union[Dict, List]) -> Union[Dict, List]:
        """Recursively delete None values from a dictionary."""
        if type(d) is dict:
            return {k: _del_none(v) for k, v in d.items() if v}
        elif type(d) is list:
            return [_del_none(v) for v in d if v]
        return d

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        try:
            if gen is None:
                return json.dumps({"status": "unhealthy", "reason": "model not loaded"}), 503

            distributed_info = {"distributed": builder_args.distributed}
            if builder_args.distributed:
                distributed_info.update({
                    "world_size": dist.get_world_size(),
                    "rank": rank,
                    "tp": builder_args.tp,
                    "pp": builder_args.pp,
                })

            return json.dumps({
                "status": "healthy",
                "model": args.model if hasattr(args, 'model') else "unknown",
                **distributed_info,
                "ready": True
            }), 200

        except Exception as e:
            return json.dumps({"status": "unhealthy", "reason": str(e)}), 503

    @app.route(f"/{OPENAI_API_VERSION}/chat/completions", methods=["POST"])
    def chat_endpoint():
        """Chat completion endpoint."""
        print(" === Completion Request ===")

        # Parse request
        data = request.get_json()
        req = CompletionRequest(**data)

        if seed := request.args.get("seed"):
            torch.manual_seed(int(seed))

        # Broadcast to other ranks if distributed
        if builder_args.distributed and dist.get_world_size() > 1:
            # Ensure all ranks are ready
            dist.barrier()

            # Broadcast the request
            req_list = [req]
            dist.broadcast_object_list(req_list, src=0)

        if req.stream:
            def chunk_processor(chunked_completion_generator):
                """Process and yield chunks."""
                for chunk in chunked_completion_generator:
                    if (next_tok := chunk.choices[0].delta.content) is None:
                        next_tok = ""
                    print(next_tok, end="", flush=True)
                    yield f"data:{json.dumps(_del_none(asdict(chunk)))}\n\n"
                print()  # New line after completion

            resp = Response(
                chunk_processor(gen.chunked_completion(req)),
                mimetype="text/event-stream",
            )
            return resp
        else:
            response = gen.sync_completion(req)
            return json.dumps(_del_none(asdict(response)))

    @app.route(f"/{OPENAI_API_VERSION}/models", methods=["GET"])
    def models_endpoint():
        return json.dumps(asdict(get_model_info_list(args)))

    @app.route(f"/{OPENAI_API_VERSION}/models/<model_id>", methods=["GET"])
    def models_retrieve_endpoint(model_id):
        if response := retrieve_model_info(args, model_id):
            return json.dumps(asdict(response))
        else:
            return "Model not found", 404

    return app


def main(args):
    """Main entry point for the server."""
    # Check if distributed
    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))

    if is_distributed and rank != 0:
        # Non-zero ranks create app (which enters worker loop)
        create_app(args)
    else:
        # Rank 0 runs Flask
        app = create_app(args)

        # Set up shutdown handler
        if is_distributed:
            def shutdown_handler():
                # Broadcast shutdown to other ranks
                req_list = ["SHUTDOWN"]
                for dst_rank in range(1, dist.get_world_size()):
                    try:
                        dist.broadcast_object_list(req_list, src=0)
                    except:
                        pass
                if dist.is_initialized():
                    dist.destroy_process_group()

            atexit.register(shutdown_handler)

        # Run Flask server
        port = args.port
        print(f"Starting server on rank {rank}, port {port}")

        try:
            app.run(host='0.0.0.0', port=port)
        except KeyboardInterrupt:
            print("Server shutting down...")
            if is_distributed and dist.is_initialized():
                # Send shutdown signal
                req_list = ["SHUTDOWN"]
                for dst_rank in range(1, dist.get_world_size()):
                    try:
                        dist.broadcast_object_list(req_list, src=0)
                    except:
                        pass