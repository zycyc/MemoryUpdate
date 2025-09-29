# MemoryUpdate Training Workflow

## Data Preprocessing
**preprocess_locomo.py**: Creates base namespace for each QA pair (e.g., `conv-26-qa159-4f01449a`)
**precompute_embeddings.py**: Precomputes embeddings for each memory

## Validation Phase
**Location**: `_validate()` in `verl/verl/trainer/ppo/ray_trainer.py:500`

1. **Prepare batch data** (`ray_trainer.py:511-517`)
   - Load validation data from dataloader (line 511)
   - Create DataProto from test data (line 512)
   - Repeat batch based on `val_kwargs.n` parameter (line 515-517)

2. **Generate sequences** (`ray_trainer.py:553-555`)
   - For each validation trial:

     a. **Run agent loop** (`tool_agent_loop.py:65`)
        - Execute tool agent loop with `async def run()`
          1. Transform namespace from base to trial-level (adds request_id, line 69)
          2. Initialize trial-specific memory store via MemoryStoreManager (line 108)
          3. Generate LLM output with tool calling support (lines 80-120)
          4. Extract and execute tool calls (max_parallel_calls, line 47/157)
          5. Execute tool with kwargs (line 274)

     b. **Compute reward score** (`memory_reward.py:143`)
        - Main entry point: `__call__()` method
        - Retrieve initial and final memories from MemoryStoreManager
        - Calculate performance delta using RAG retrieval
        - Apply memory efficiency factors

     c. **Pad sequences** (`agent_loop.py:148-160`)
        - AgentLoopOutput dataclass defines padded fields
        - Pad prompt to fixed length
        - Pad response to fixed length

     d. **Postprocess** (`agent_loop.py:601`)
        - `_postprocess()` method processes padded outputs from agent loop
        - Combine into batch tensors

3. **Report metrics**
   - Log validation scores and sample outputs
   - Track memory growth statistics

## Training Phase
**Location**: Main training loop in `verl/verl/trainer/ppo/ray_trainer.py`

For each epoch (`range(self.config.trainer.total_epochs)`, line 952):

  For each batch in train_dataloader (line 953):

    1. **Generate sequences** (`ray_trainer.py:991-993`)
       - Same flow as validation phase
       - Calls `generate_sequences()` on actor_rollout_wg or async_rollout_manager
       - Includes reward computation inline (line 1045)
       - Tracks conversation statistics via MemoryStoreManager

    2. **Compute log probabilities** (`ray_trainer.py:1048-1058`)
       - Recompute old_log_probs using actor model (line 1050)
       - Extract entropy for regularization (line 1051-1054)
       - Update batch with old_log_prob (line 1058)

    3. **Compute values** (if using critic)
       - Get value estimates from critic network

    4. **Compute advantages** (`ray_trainer.py` calls `compute_advantage()`)
       - Function defined at line 198 in ray_trainer.py
       - Calculate GAE or other advantage estimation (line 230-231)
       - Balance valid tokens across DP ranks if enabled

    5. **Periodic validation**
       - Run validation every N steps (same as validation phase above)

    6. **Update metrics and logging**
       - Log to WandB/TensorBoard
       - Save checkpoints based on save_freq
       - Track timing and performance metrics

## Key Components

### Namespace Management
- **Base namespace**: Created during preprocessing (e.g., `conv-26-qa159-4f01449a`)
- **Trial namespace**: Base + request_id suffix for isolation (e.g., `conv-26-qa159-4f01449a-12345678`)

### Memory Flow
1. Initial memories loaded from preprocessed data
2. Tools modify memory during agent loop execution
3. MemoryStoreManager tracks changes via Ray Actor
4. Reward computed based on memory state delta

### Tool Execution
- Tools created/destroyed per execution via verl's tool system
- Memory operations proxied through MemoryBrokerActor
- Maximum 1 parallel tool call (configurable)
- Tool responses truncated to max_tool_response_length