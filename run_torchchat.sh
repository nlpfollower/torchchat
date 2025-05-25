#!/usr/bin/bash

set -ex

NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
torchchat.py $overrides