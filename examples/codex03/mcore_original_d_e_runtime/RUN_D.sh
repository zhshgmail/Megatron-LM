#!/bin/bash
# Original training arguments preserved. Full D/E model run remains NOT_INVOKED for this candidate.
# Fresh original initialization; uninterrupted 100 updates. Not matched-initial-state proof.
set -eu
: "${DATA_PATH:?UNBOUND_CANONICAL_WIKITEXT103_INDEXED_SPLITS_AND_GPT2_BPE}"
: "${TENSORBOARD_PATH:?UNBOUND_NEW_OWN_TENSORBOARD_OUTPUT}"
: "${CHECKPOINT_SAVE_PATH:?UNBOUND_NEW_OWN_SAVE_OUTPUT}"
: "${CHECKPOINT_LOAD_PATH:?UNBOUND_FRESH_EMPTY_INITIAL_LOAD_DIRECTORY}"
: "${DATA_CACHE_PATH:?UNBOUND_NEW_OWN_DATA_CACHE}"
: "${CANN_VENDOR_PYTHONPATH:?UNBOUND_ACTUAL_VENDOR_PYTHONPATH}"
: "${MASTER_PORT:?UNBOUND_PRIVATE_PORT}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NCCL_ALGO=Ring CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
unset PYTHONHOME
MCORE_SOURCE_ROOT="${MCORE_SOURCE_ROOT:?UNBOUND_SELECTED_MCORE_SOURCE}"
PY="${PYTHON:?UNBOUND_ACTUAL_COMPATIBLE_INTERPRETER}"
source "$(dirname "${BASH_SOURCE[0]}")/selected_runtime.sh"
# Retain actual CANN/driver shared-library paths from the bound image; no guessed LD_LIBRARY_PATH.
# Selected source533cf pretrain_gpt.py already imports the original megatron_adaptor.
cd "$MCORE_SOURCE_ROOT"
"$PY" -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
 "${TRAINING_OBSERVER:?UNBOUND_ORIGINAL_OBSERVER}" "${STANDALONE_ENTRY:?UNBOUND_ORIGINAL_STANDALONE_ENTRY}" \
 --num-layers 12 \
 --hidden-size 512 \
 --num-attention-heads 8 \
 --log-params-norm \
 --log-num-zeros-in-grad \
 --log-validation-ppl-to-tensorboard \
 --log-timers-to-tensorboard \
 --tensorboard-dir "${TENSORBOARD_PATH}" \
 --micro-batch-size 4 \
 --global-batch-size 32 \
 --seq-length 1024 \
 --max-position-embeddings 1024 \
 --train-iters 100 --dataloader-worker-seed 1234 \
 --timing-log-level 0 \
 --lr-decay-iters 320000 \
 --save "${CHECKPOINT_SAVE_PATH}" \
 --load "${CHECKPOINT_LOAD_PATH}" \
 --train-data-path "${DATA_PATH}/indexed/train_text_document" \
 --valid-data-path "${DATA_PATH}/indexed/validation_text_document" \
 --test-data-path "${DATA_PATH}/indexed/test_text_document" \
 --vocab-file "${DATA_PATH}/tokenizer/vocab.json" \
 --merge-file "${DATA_PATH}/tokenizer/merges.txt" \
 --distributed-backend nccl \
 --lr 0.00015 \
 --lr-decay-style cosine \
 --min-lr 1.0e-5 \
 --weight-decay 1e-2 \
 --clip-grad 1.0 \
 --lr-warmup-fraction .01 \
 --log-interval 1 \
 --save-interval 50 \
 --eval-interval 1000 \
 --eval-iters 10 \
 --transformer-impl transformer_engine \
 --tensor-model-parallel-size 4 \
 --pipeline-model-parallel-size 1 \
 --use-distributed-optimizer \
 --overlap-grad-reduce \
 --overlap-param-gather \
 --deterministic-mode \
 --no-gradient-accumulation-fusion \
 --attention-softmax-in-fp32 \
 --use-mcore-models \
 --ckpt-format torch_dist \
 --dist-ckpt-strictness log_all \
 --data-cache-path "${DATA_CACHE_PATH}" \
 --bf16 \
 --attention-backend unfused \
 --log-memory-to-tensorboard
