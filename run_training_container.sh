#!/bin/bash

# MemUpdate Training Script - Container version
# Run from within the verl Docker container

# TODO: check
# why is num_turns/max 14 when i set max_assistant_turns to be 30? what's the stop condition?

# TODO: Update model path below (line 63) with your downloaded Qwen3-8B snapshot ID
# See README.md "Required HuggingFace Models" section for instructions on:
#   1. Downloading Qwen3-8B and Qwen3-Embedding-0.6B models
#   2. Finding the correct snapshot path in /root/.cache/huggingface/hub/

set -e

ulimit -n 65535

# Load environment variables from .env file if it exists
if [ -f "/workspace/memupdate/.env" ]; then
    echo "📋 Loading environment variables from .env file..."
    export $(grep -v '^#' /workspace/memupdate/.env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  Warning: .env file not found at /workspace/memupdate/.env"
    echo "   Copy .env.example to .env and configure your settings"
fi

# Verify required environment variables
if [ -z "$LITELLM_API_KEY" ]; then
    echo "❌ Error: LITELLM_API_KEY not set. Please configure it in .env file."
    exit 1
fi

if [ -z "$EVALUATOR_URL" ]; then
    echo "❌ Error: EVALUATOR_URL not set. Please configure it in .env file."
    exit 1
fi

# Optional: Check WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set. Training logs will not be uploaded to W&B."
fi

# Prevent HuggingFace from trying to connect online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Container paths
PROJECT_DIR="/workspace/memupdate/verl"  # Use submodule verl
MEMUPDATE_DIR="/workspace/memupdate"

cd $PROJECT_DIR

echo "Starting MemUpdate training from verl directory..."
echo "Project directory: $PROJECT_DIR"
echo "MemUpdate directory: $MEMUPDATE_DIR"

# Check if data is preprocessed
if [ ! -f "$MEMUPDATE_DIR/data/locomo/train.parquet" ]; then
    echo "Training data not found. Running preprocessing..."
    cd $MEMUPDATE_DIR && python -m memupdate.data.preprocess_locomo
    cd $PROJECT_DIR
fi

# Run verl training using existing GSM8K config as base
echo "Starting GRPO training with verl..."

export PYTHONPATH="/workspace/memupdate/verl:/workspace/memupdate:$PYTHONPATH"

# Ensure memupdate is imported for reward manager registration
python3 -c "import memupdate; print('✅ MemoryRewardManager registered')"

echo "🚀 Starting training..."

# Run training with Ray runtime environment for worker import
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/sglang_multiturn/config" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=wandb \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='memupdate-rl' \
    trainer.experiment_name='qwen3-8b-memory-grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=9999999 \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$MEMUPDATE_DIR/config/memory_tools.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
    data.train_files="$MEMUPDATE_DIR/data/locomo/train.parquet" \
    data.val_files="$MEMUPDATE_DIR/data/locomo/test.parquet" \
    trainer.total_epochs=100 \
    trainer.log_train_generations=5 \
    trainer.log_train_freq=5 \
    reward_model.reward_manager=memory_rag \
    memupdate.memory_broker_shards=32 \


echo "📋 Training completed!"