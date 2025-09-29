"""MemUpdate: Self-Refining Memory via Reinforcement Learning for LLM Agents."""

__version__ = "0.1.0"

# Register custom reward manager with verl
try:
    from verl.workers.reward_manager import register
    from .rewards.memory_reward import MemoryRewardManager
    
    # Use register as decorator
    MemoryRewardManager = register("memory_rag")(MemoryRewardManager)
    print("✅ Registered MemoryRewardManager with verl as 'memory_rag'")
except ImportError:
    print("⚠️  verl not available - reward manager not registered")
