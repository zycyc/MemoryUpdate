# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules

1. Do NOT git commit or push unless explicitly told to do so.
2. Python execution must be done in Docker container - local Python lacks dependencies.
3. File examination and editing should be done locally (files are mounted to Docker).
4. Do NOT speculate on what the code does or how things might work or be implemented. Always thoroughly read the code before making any assumptions and research more if needed.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient and production-ready RL training library for large language models (LLMs). It's the open-source implementation of the HybridFlow framework, supporting various RL algorithms like PPO, GRPO, DAPO, and more for LLM post-training.

## Common Development Commands

### Building and Installation

```bash
# Install base requirements
pip install -e .

# Install with optional dependencies for development
pip install -e .[test,prime,geo,gpu,math,vllm,sglang,trl,mcore]

# Install specific backend dependencies
pip install -e .[vllm]   # For vLLM backend
pip install -e .[sglang]  # For SGLang backend
pip install -e .[mcore]   # For Megatron-LM backend
```

### Running Tests

```bash
# Run CPU unit tests (tests with *_on_cpu.py suffix)
pytest -s -x --asyncio-mode=auto tests/

# Run GPU unit tests (excluding special tests)
pytest -s tests/ --ignore=tests/special_e2e --ignore=tests/special_distributed --ignore=tests/special_npu

# Run specific test category
pytest -s tests/trainer/  # Test trainer components
pytest -s tests/workers/  # Test worker components
pytest -s tests/models/   # Test model components

# Run end-to-end tests (requires GPUs)
bash tests/special_e2e/run_test.sh
```

### Linting and Code Quality

```bash
# Run ruff linter
ruff check .
ruff format .

# Type checking
mypy verl/

# Pre-commit hooks
pre-commit run --all-files
```

### Training Examples

```bash
# Run GRPO training with Qwen model
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=path/to/data.parquet \
    actor_rollout_ref.model.path=path/to/model \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1

# Run PPO training
python3 -m verl.trainer.main_ppo \
    data.train_files=path/to/data.parquet \
    actor_rollout_ref.model.path=path/to/model

# Run SFT (Supervised Fine-Tuning)
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=path/to/data.parquet \
    model.path=path/to/model
```

## Code Architecture

### Core Components

1. **verl/trainer/**: Main training logic and orchestration
   - `main_ppo.py`: PPO training entry point
   - `fsdp_sft_trainer.py`: SFT training with FSDP
   - Configuration managed via Hydra with YAML configs in `config/`

2. **verl/workers/**: Distributed worker implementations
   - `actor/`: Actor model workers for policy updates
   - `critic/`: Critic model workers for value estimation
   - `rollout/`: Rollout workers for generation (vLLM, SGLang, HF)
   - `reward_model/`: Reward model workers

3. **verl/single_controller/**: Ray-based distributed control
   - Manages worker groups and resource allocation
   - Handles communication between different components

4. **verl/models/**: Model definitions and adaptations
   - Support for various architectures (Qwen, Llama, DeepSeek, etc.)
   - Integration with Megatron-LM and HuggingFace models

5. **verl/utils/**: Utility functions
   - `dataset/`: Data loading and processing utilities
   - `checkpoint/`: Checkpoint saving/loading
   - `reward_score/`: Reward calculation functions
   - `megatron/`: Megatron-specific utilities

### Training Flow

1. **Data Preparation**: Load and preprocess data from parquet files
2. **Model Initialization**: Initialize actor, critic, reference, and reward models
3. **Rollout Generation**: Generate responses using vLLM/SGLang
4. **Reward Calculation**: Compute rewards using reward models or functions
5. **Policy Update**: Update actor model using PPO/GRPO/DAPO algorithms
6. **Logging & Checkpointing**: Track metrics and save model checkpoints

### Configuration System

verl uses Hydra for configuration management. Key configuration files:
- `verl/trainer/config/ppo_trainer.yaml`: PPO training config
- `verl/trainer/config/sft_trainer.yaml`: SFT training config
- Configs can be overridden via command line arguments

### Backend Support

- **FSDP/FSDP2**: Default distributed training backend
- **Megatron-LM**: For very large models (>100B parameters)
- **vLLM**: High-throughput inference engine
- **SGLang**: Supports multi-turn and tool-calling scenarios

## Important Considerations

### Resource Management
- GPU memory utilization can be controlled via `rollout.gpu_memory_utilization`
- Use gradient checkpointing for larger models: `enable_gradient_checkpointing=True`
- FSDP offloading options available for memory-constrained scenarios

### Multi-GPU and Multi-Node
- Use `trainer.n_gpus_per_node` and `trainer.nnodes` to configure distributed training
- Ray handles resource allocation and worker management
- Support for tensor parallelism and pipeline parallelism

### Data Format
- Training data should be in parquet format
- Expected columns: `data_source`, `prompt`, `response`, `messages` (for chat format)
- Use preprocessing scripts in `examples/data_preprocess/` for data preparation

### Experiment Tracking
- Supports WandB, TensorBoard, MLflow for logging
- Configure via `trainer.logger` parameter
- Checkpoints saved according to `trainer.save_freq`

## Debugging Tips

1. Set environment variables for debugging:
   ```bash
   export HYDRA_FULL_ERROR=1  # Show full error traces
   export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA operations
   ```

2. Use diagnostic script for environment check:
   ```bash
   python scripts/diagnose.py
   ```

3. Check Ray cluster status:
   ```bash
   ray status
   ```

4. For memory issues, enable CPU offloading:
   ```bash
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   ```