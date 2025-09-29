"""Shared memory store management for all tools."""

import ray
import os
import logging
import asyncio
# torch not needed for CPU-only embeddings
from typing import Dict, List, Optional, Any

try:
    from langgraph.store.memory import InMemoryStore
    # IndexConfig not needed for CPU-only cached embeddings
except ImportError:
    InMemoryStore = None

logger = logging.getLogger(__name__)

# Maximum results to return when list_all=True (effectively unlimited)
UNLIMITED_RESULTS = 10000


@ray.remote(num_cpus=1)  # No GPU allocation - shares resources with verl workers
class MemoryBrokerActor:
    """
    Ray Actor that serves as memory broker between rollout workers and reward workers.
    
    Key Design:
    - Shared across ALL Ray workers in the verl cluster
    - Manages InMemoryStore instances per conversation trial_namespace
    - Handles concurrent access from rollout workers (write) and reward workers (read)
    - Provides memory isolation per batch item
    """
    
    def __init__(self):
        # trial_namespace -> InMemoryStore instance
        self._stores: Dict[str, InMemoryStore] = {}
        
        # Centralized storage for conversation memories by sample_id
        # sample_id (e.g., "conv-48") -> initial memories from LoCoMo dataset
        self._conversation_memories: Dict[str, List[Dict]] = {}
        
        # Memory ID counters per trial_namespace for agent-friendly IDs
        self._memory_counters: Dict[str, int] = {}
        
        # Cached embeddings (key: content hash, value: embedding with metadata)
        self._embedding_cache = {}
        self._load_embedding_cache()
        self._load_conversation_memories()
        
        
        # MEMUPDATE: Registry of rollout workers with embedding models
        self._rollout_embedding_models: Dict[str, Any] = {}  # worker_id -> worker_ray_handle
        
        # MEMUPDATE: Round-robin counter for worker selection
        self._worker_round_robin = 0
        
        print(f"🏢 MemoryBrokerActor initialized with {len(self._conversation_memories)} conversations in process {os.getpid()}")
    
    def _load_embedding_cache(self):
        """Load cached embeddings with metadata."""
        try:
            import pickle
            import numpy as np
            
            cache_file = "/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl"
            
            with open(cache_file, 'rb') as f:
                self._embedding_cache = pickle.load(f)
            
            # Validate embeddings
            for key, value in self._embedding_cache.items():
                embedding = value.get('embedding')
                if isinstance(embedding, np.ndarray):
                    norm = np.linalg.norm(embedding)
                    if norm == 0 or np.isnan(norm):
                        print(f"⚠️ Bad embedding for key {key}: norm={norm}")
                        value['embedding'] = np.random.randn(1024) * 0.01
            
            # Count embeddings per conversation
            conv_counts = {}
            for key, value in self._embedding_cache.items():
                sid = value.get('sample_id')
                if sid:
                    conv_counts[sid] = conv_counts.get(sid, 0) + 1
            
            print(f"💾 Loaded {len(self._embedding_cache)} embeddings for {len(conv_counts)} conversations")
            if conv_counts:
                print(f"📊 Sample distribution: {list(conv_counts.items())[:3]}...")
                
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")
            self._embedding_cache = {}
    
    def _load_conversation_memories(self):
        """Load all conversation memories from LoCoMo dataset at startup."""
        try:
            import json
            from pathlib import Path
            
            locomo_path = Path("/workspace/memupdate/data/locomo10.json")
            if not locomo_path.exists():
                print(f"⚠️ LoCoMo data not found at {locomo_path}, skipping conversation memory loading")
                return
            
            with open(locomo_path, 'r') as f:
                locomo_data = json.load(f)
            
            # Process each conversation with all data levels enabled by default
            for conv in locomo_data:
                sample_id = conv.get("sample_id", "unknown")
                
                # Convert all data types to memory format
                memories = self._convert_all_data_to_memories(
                    conv,
                    include_observation=True,
                    include_conversation=True,
                    include_event_summary=True
                )
                
                self._conversation_memories[sample_id] = memories
                
                # Count by source type for logging
                source_counts = {}
                for mem in memories:
                    source = mem.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                print(f"  📚 Loaded {len(memories)} memories for {sample_id}: {source_counts}")
            
            print(f"✅ Loaded conversation memories for {len(self._conversation_memories)} conversations")
            
        except Exception as e:
            print(f"❌ Failed to load conversation memories: {e}")
            import traceback
            traceback.print_exc()
    
    def _convert_all_data_to_memories(self, conv_data: Dict, 
                                     include_observation: bool = True,
                                     include_conversation: bool = True,
                                     include_event_summary: bool = True) -> List[Dict]:
        """Convert all data types from conversation to memory entries.
        
        Processes three levels of data:
        1. observation - Extracted facts from conversations
        2. conversation - Raw dialogue turns  
        3. event_summary - High-level event summaries
        """
        memories = []
        
        # Extract session timestamps from conversation level
        timestamps = {}
        conversation_data = conv_data.get('conversation', {})
        for key in conversation_data.keys():
            if key.endswith('_date_time'):
                session_num = key.replace('_date_time', '')
                timestamps[session_num] = conversation_data[key]
        
        # 1. Process observations (extracted facts)
        if include_observation and 'observation' in conv_data:
            observations = conv_data['observation']
            for session_key, session_obs in observations.items():
                if isinstance(session_obs, dict):
                    # Extract session number (e.g., "session_1" from "session_1_observation")
                    session_num = session_key.replace('_observation', '')
                    timestamp = timestamps.get(session_num, "unknown time")
                    
                    for speaker, speaker_observation in session_obs.items():
                        for fact_entry in speaker_observation:
                            if isinstance(fact_entry, list) and len(fact_entry) >= 2:
                                fact_text, evidence = fact_entry[0], fact_entry[1]
                                
                                memory = {
                                    "content": fact_text,
                                    "speaker": speaker,
                                    "evidence": evidence,
                                    "session": session_num,  # Just "session_1", not "session_1_observation"
                                    "timestamp": timestamp,
                                    "memory_type": "episodic",
                                    "source": "observation",
                                }
                                memories.append(memory)
        
        # 2. Process conversation dialogue
        if include_conversation and 'conversation' in conv_data:
            for session_key, dialogue_data in conversation_data.items():
                if session_key.startswith('session_') and not session_key.endswith('_date_time'):
                    timestamp = timestamps.get(session_key, "unknown time")
                    
                    if isinstance(dialogue_data, list):
                        for turn in dialogue_data:
                            if isinstance(turn, dict) and 'text' in turn:
                                memory = {
                                    "content": turn.get('text', ''),
                                    "speaker": turn.get('speaker', ''),
                                    "evidence": turn.get('dia_id', ''),
                                    "session": session_key,  # "session_1", "session_2", etc.
                                    "timestamp": timestamp,
                                    "memory_type": "episodic",
                                    "source": "conversation"
                                }
                                memories.append(memory)
        
        # 3. Process event summaries
        if include_event_summary and 'event_summary' in conv_data:
            events = conv_data['event_summary']
            for event_key, event_data in events.items():
                if isinstance(event_data, dict):
                    # Extract session number from "events_session_1" -> "session_1"
                    session_num = event_key.replace('events_', '')
                    # Use conversation timestamp if available, else use event date
                    timestamp = timestamps.get(session_num, event_data.get('date', 'unknown time'))
                    
                    for speaker, event_list in event_data.items():
                        if speaker != 'date' and isinstance(event_list, list):
                            for event in event_list:
                                if event:  # Skip empty events
                                    memory = {
                                        "content": event,
                                        "speaker": speaker,
                                        "evidence": event_key,
                                        "session": session_num,  # "session_1", "session_2", etc.
                                        "timestamp": timestamp,
                                        "memory_type": "episodic",
                                        "source": "event_summary"
                                    }
                                    memories.append(memory)
        
        return memories
    
    def _convert_observation_to_memories(self, observations: Dict) -> List[Dict]:
        """Legacy method - kept for compatibility but delegates to new method."""
        # Create a minimal conv_data structure with just observations
        conv_data = {'observation': observations}
        return self._convert_all_data_to_memories(
            conv_data, 
            include_observation=True,
            include_conversation=False,
            include_event_summary=False
        )
    
    def register_embedding_model(self, worker_id: str):
        """MEMUPDATE: Register rollout worker ID for embedding access."""
        # Just store that this worker has embedding models available
        self._rollout_embedding_models[worker_id] = True
        print(f"🔗 Registered embedding worker {worker_id} (total: {len(self._rollout_embedding_models)})")
        
    def get_available_embedding_worker_id(self) -> str:
        """MEMUPDATE: Get a worker ID for a rollout worker with embedding model using round-robin."""
        if not self._rollout_embedding_models:
            print(f"⚠️ No rollout workers with embedding models available")
            return None
            
        # Use round-robin to distribute load across all workers
        worker_ids = list(self._rollout_embedding_models.keys())
        worker_id = worker_ids[self._worker_round_robin % len(worker_ids)]
        self._worker_round_robin += 1
        # print(f"📡 Providing rollout worker {worker_id} for direct embedding access (RR position: {self._worker_round_robin}/{len(worker_ids)})")
        return worker_id
    
    def get_initial_memories(self, sample_id: str) -> List[Dict]:
        """Get initial memories for a sample_id from loaded conversation data."""
        memories = self._conversation_memories.get(sample_id, [])
        if not memories:
            print(f"⚠️ No conversation memories found for sample_id '{sample_id}'")
        return memories.copy()  # Return a copy to prevent modification
    
    def _generate_agent_friendly_id(self, trial_namespace: str) -> str:
        """Generate simple sequential memory ID that can be reliably referenced and updated."""
        # Initialize counter for this namespace if not exists
        if trial_namespace not in self._memory_counters:
            self._memory_counters[trial_namespace] = 0
        
        # Increment counter
        self._memory_counters[trial_namespace] += 1
        counter = self._memory_counters[trial_namespace]
        
        # Use simple sequential ID: memory_1, memory_2, etc.
        return f"memory_{counter}"
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded conversations for verification."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
        except:
            memory_mb = 0
            
        return {
            "total_conversations": len(self._conversation_memories),
            "sample_ids": list(self._conversation_memories.keys()),
            "memory_counts": {sid: len(mems) for sid, mems in self._conversation_memories.items()},
            "active_stores": len(self._stores),
            "memory_usage_mb": memory_mb,
            "store_namespaces": list(self._stores.keys())[:10]  # Show first 10 to avoid huge output
        }
    
    
    def _get_sample_id_from_namespace(self, trial_namespace: str) -> str:
        """Extract sample_id from trial_namespace (e.g., 'conv-48-qa2-abc123' -> 'conv-48')."""
        if '-qa' in trial_namespace:
            return trial_namespace.split('-qa')[0]
        return trial_namespace
    
    def _create_store_with_embeddings(self, trial_namespace: str) -> InMemoryStore:
        """Create InMemoryStore with smart cached embeddings."""
        try:
            from memupdate.tools.cached_embeddings import SmartCachedEmbeddings
            import torch
            
            # Extract sample_id from trial_namespace for filtering
            sample_id = self._get_sample_id_from_namespace(trial_namespace)
            
            # Create smart embeddings that:
            # 1. Uses our already-loaded cache
            # 2. Filters by conversation  
            # 3. Uses rollout worker for direct GPU embedding access
            embedding_worker_id = self.get_available_embedding_worker_id()
            embeddings = SmartCachedEmbeddings(
                cache=self._embedding_cache,  # Use already-loaded cache
                sample_id=sample_id,  # Filter by conversation
                embedding_worker_id=embedding_worker_id,  # Use rollout worker ID
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            index_config = {
                "embed": embeddings,
                "dims": 1024,  # Qwen3-0.6B dimension
                "fields": ["content"],
            }
            
            store = InMemoryStore(index=index_config)
            
            return store
            
        except Exception as e:
            print(f"❌ Failed to create store with embeddings: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic store
            store = InMemoryStore()
            print(f"⚠️ Created basic InMemoryStore (no embeddings) for trial_namespace '{trial_namespace}'")
            return store
    
    async def create_memory_in_store(self, trial_namespace: str, memory_data: dict) -> dict:
        """Create memory with agent-friendly IDs using direct store operations."""
        
        if trial_namespace not in self._stores:
            store = self._create_store_with_embeddings(trial_namespace)
            self._stores[trial_namespace] = store
        
        store = self._stores[trial_namespace]
        
        try:
            # Generate agent-friendly memory ID if not provided
            if "id" in memory_data:
                memory_id = memory_data["id"]
            else:
                # Generate sequential ID
                memory_id = self._generate_agent_friendly_id(trial_namespace)
            
            # Prepare memory value with robust metadata handling
            metadata_raw = memory_data.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            
            memory_value = {
                "content": memory_data.get("content", ""),
                "metadata": metadata,
                "id": memory_id
            }
            
            # Store with agent-friendly ID that agent can reference
            await store.aput(
                ("memories",),
                memory_id,
                memory_value
            )
            
            return {
                "result": f"Created memory with ID: {memory_id}",
                "success": True,
                "memory_id": memory_id
            }
            
        except Exception as e:
            print(f"❌ Error creating memory in Ray Actor store: {e}")
            return {
                "result": str(e),
                "success": False
            }
    
    async def update_memory_in_store(self, trial_namespace: str, memory_data: dict) -> dict:
        """Update memory using agent-friendly ID directly (no conversion needed)."""
        
        if trial_namespace not in self._stores:
            # Create store with embeddings if it doesn't exist (rare case)
            print(f"🔧 Creating new store with embeddings for trial_namespace '{trial_namespace}' during update")
            store = self._create_store_with_embeddings(trial_namespace)
            self._stores[trial_namespace] = store
        
        store = self._stores[trial_namespace]
        
        try:
            # Get memory ID - required for updates, use directly as agent provided it
            memory_id = memory_data.get("id")
            if not memory_id:
                return {
                    "result": "Memory ID required for updates",
                    "success": False
                }
            
            # Get existing memory to preserve metadata
            try:
                existing = await store.aget(("memories",), memory_id)
                if existing:
                    existing_metadata = existing.value.get("metadata", {})
                    # zycyc debug prints
                    # print(f"✅ Found existing memory to update: {memory_id}")
                    is_actual_update = True
                else:
                    existing_metadata = {}
                    print(f"⚠️ Memory ID {memory_id} not found, will create new memory instead")
                    if memory_id.startswith('mem_') or len(memory_id) > 10:
                        print(f"   💡 Hint: Use search_memory() first to get valid IDs")
                    is_actual_update = False
            except Exception as e:
                existing_metadata = {}
                print(f"⚠️ Error retrieving existing memory {memory_id}, will create new memory: {e}")
                is_actual_update = False
            
            # Prepare updated memory value with robust metadata handling
            new_metadata_raw = memory_data.get("metadata", {})
            new_metadata = new_metadata_raw if isinstance(new_metadata_raw, dict) else {}
            
            memory_value = {
                "content": memory_data.get("content", ""),
                "metadata": {**existing_metadata, **new_metadata},
                "id": memory_id
            }
            
            # Direct store update operation using agent-friendly ID
            await store.aput(
                ("memories",),
                memory_id,
                memory_value
            )
            
            # Return appropriate message based on whether it was update or creation
            if is_actual_update:
                result_message = f"Successfully updated existing memory: {memory_id}"
            else:
                result_message = f"Created new memory (ID not found): {memory_id}"
            
            return {
                "result": result_message,
                "success": True,
                "memory_id": memory_id,
                "was_actual_update": is_actual_update
            }
            
        except Exception as e:
            print(f"❌ Error updating memory in Ray Actor store: {e}")
            return {
                "result": str(e),
                "success": False
            }
    
    async def search_memory_in_store(self, trial_namespace: str, query: str, limit: Optional[int] = 5, source_filter: str = "", speaker_filter: str = "", session_filter: str = "", list_all: bool = False) -> dict:
        """Search memory directly in Ray Actor store with multi-field filtering."""
        
        if trial_namespace not in self._stores:
            print(f"⚠️ Namespace '{trial_namespace}' not found in stores for search memory")
            return {"results": [], "success": True}
        
        store = self._stores[trial_namespace]
        
        try:
            # Handle list_all mode
            if list_all:
                # List all mode: use high limit and no query for temporal ordering
                actual_limit = UNLIMITED_RESULTS
                actual_query = None  # None query preserves temporal order
            else:
                # Normal mode: use provided limit or default
                actual_limit = limit if limit is not None else 5
                actual_query = query
            
            # Direct store search - follows mem0 pattern: memories = await store.asearch(query=question, limit=5)
            has_embeddings = hasattr(store, 'index_config') and store.index_config is not None
            
            # Use comprehensive multi-value filtering for all fields
            filter_dict = None
            
            # Parse and validate all filter fields
            source_list = self._parse_multi_value_filter(source_filter, {"conversation", "observation", "event_summary", "agent_output"}, "")
            speaker_list = self._parse_multi_value_filter(speaker_filter, set(), "")  # No validation set for speakers
            session_list = self._parse_multi_value_filter(session_filter, set(), "")  # No validation set for sessions
            
            # Check if we need multi-value search
            needs_multi_search = (
                len(source_list) > 1 or 
                len(speaker_list) > 1 or 
                len(session_list) > 1
            )
            
            if needs_multi_search:
                # Use comprehensive multi-value search
                return await self._multi_value_search(trial_namespace, actual_query, actual_limit, has_embeddings, source_list, speaker_list, session_list, list_all)
            
            # Single-value filtering - use direct search
            metadata_filters = {}
            if source_list and source_list != ["all"]:
                metadata_filters["source"] = source_list[0]
            if speaker_list:
                metadata_filters["speaker"] = speaker_list[0]
            if session_list:
                metadata_filters["session"] = session_list[0]
                
            if metadata_filters:
                filter_dict = {"metadata": metadata_filters}

            memories = await store.asearch(
                ("memories",),
                query=actual_query,
                filter=filter_dict,
                limit=actual_limit
            )

            # Simple result formatting - capture SearchItem.score for consistency
            results = []
            for mem in memories:
                results.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                    "score": mem.score  # Capture relevance/similarity score
                })
            
            return {
                "results": results,
                "success": True,
                "semantic_search": has_embeddings,
                "source_filter": source_filter
            }
            
        except Exception as e:
            print(f"❌ Error searching memory in Ray Actor store: {e}")
            import traceback
            traceback.print_exc()
            return {
                "results": [],
                "success": False
            }
    
    def _parse_multi_value_filter(self, filter_value: str, valid_values: set, default_ignore: str) -> list:
        """Parse comma-separated filter values and validate them."""
        if not filter_value or filter_value.lower() in ["", "any", default_ignore]:
            return []
        
        # Split by comma and clean up
        values = [v.strip() for v in filter_value.split(",") if v.strip()]
        
        # Validate against valid_values if provided
        if valid_values:
            values = [v for v in values if v in valid_values]
        
        return values

    async def _multi_value_search(self, trial_namespace: str, query: str, limit: int, has_embeddings: bool, source_list: list, speaker_list: list, session_list: list, list_all: bool = False) -> dict:
        """Perform comprehensive multi-value search across all field combinations."""
        try:
            # Generate all combinations of filter values
            combinations = self._generate_filter_combinations(source_list, speaker_list, session_list)
            
            # Limit combinations to prevent excessive parallel searches
            max_combinations = 20  # Reasonable limit for performance
            if len(combinations) > max_combinations:
                print(f"⚠️ Too many filter combinations ({len(combinations)}), limiting to {max_combinations}")
                combinations = combinations[:max_combinations]
            
            # Run parallel searches for each combination
            search_tasks = []
            for filters in combinations:
                search_tasks.append(self._search_single_combination(trial_namespace, query, limit, filters))
            
            # Execute all searches in parallel
            search_results = await asyncio.gather(*search_tasks)
            
            # Merge and deduplicate results
            all_memories = []
            seen_ids = set()
            
            for result in search_results:
                if result["success"]:
                    for memory in result["results"]:
                        memory_id = memory.get("id")
                        if memory_id not in seen_ids:
                            seen_ids.add(memory_id)
                            all_memories.append(memory)
            
            # Sort and limit based on mode
            if list_all and not query:
                # List all mode with no query: preserve temporal order, no limit
                final_results = all_memories
            elif has_embeddings and all_memories and query:
                # Semantic search: sort by relevance score
                all_memories.sort(key=lambda m: m.get("score") or 0.0, reverse=True)
                final_results = all_memories if list_all else all_memories[:limit]
            else:
                # Keyword search or no embeddings: preserve original order
                final_results = all_memories if list_all else all_memories[:limit]
            
            # Create filter description for logging
            filter_desc = []
            if source_list:
                filter_desc.append(f"source: {','.join(source_list)}")
            if speaker_list:
                filter_desc.append(f"speaker: {','.join(speaker_list)}")
            if session_list:
                filter_desc.append(f"session: {','.join(session_list)}")
            
            return {
                "results": final_results,
                "success": True,
                "semantic_search": has_embeddings,
                "source_filter": ",".join(filter_desc) if filter_desc else "multi_value",
                "combinations_used": len(combinations)
            }
            
        except Exception as e:
            print(f"❌ Error in multi-value search: {e}")
            return {
                "results": [],
                "success": False
            }
    
    def _generate_filter_combinations(self, source_list: list, speaker_list: list, session_list: list) -> list:
        """Generate all combinations of filter values."""
        # Use default "all" if lists are empty
        sources = source_list if source_list else ["all"]
        speakers = speaker_list if speaker_list else [None]
        sessions = session_list if session_list else [None]
        
        combinations = []
        for source in sources:
            for speaker in speakers:
                for session in sessions:
                    filters = {}
                    if source != "all":
                        filters["source"] = source
                    if speaker:
                        filters["speaker"] = speaker
                    if session:
                        filters["session"] = session
                    combinations.append(filters)
        
        return combinations
    
    async def _search_single_combination(self, trial_namespace: str, query: str, limit: int, filters: dict) -> dict:
        """Search with a single combination of filter values."""
        try:
            if trial_namespace not in self._stores:
                return {"results": [], "success": True}
            
            store = self._stores[trial_namespace]
            
            # Create filter dict if we have filters
            filter_dict = {"metadata": filters} if filters else None
            
            memories = await store.asearch(
                ("memories",),
                query=query,
                filter=filter_dict,
                limit=limit
            )
            
            # Format results - capture SearchItem.score for proper sorting
            results = []
            for mem in memories:
                results.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                    "score": mem.score  # Capture relevance/similarity score
                })
            
            return {
                "results": results,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error searching combination {filters}: {e}")
            return {
                "results": [],
                "success": False
            }
    
    async def init_conversation_memory(self, trial_namespace: str, sample_id: str, 
                                      include_observation: bool = True,
                                      include_conversation: bool = True, 
                                      include_event_summary: bool = True) -> str:
        """Initialize memory for a conversation using sample_id (called per batch item).
        
        Args:
            trial_namespace: Unique trial_namespace for this trajectory (e.g., "conv-48-qa2-abc123")
            sample_id: Conversation identifier to load initial memories from (e.g., "conv-48")
            include_observation: Include observation-level memories
            include_conversation: Include conversation dialogue
            include_event_summary: Include event summaries
        """
        # Get initial memories from loaded conversation data
        initial_memories = self.get_initial_memories(sample_id)    
        
        # Create store with embeddings
        store = self._create_store_with_embeddings(trial_namespace)
        self._stores[trial_namespace] = store
        
        # Filter memories based on data level flags
        filtered_memories = []
        for memory in initial_memories:
            source = memory.get('source', 'observation')
            if (source == 'observation' and include_observation) or \
               (source == 'conversation' and include_conversation) or \
               (source == 'event_summary' and include_event_summary):
                filtered_memories.append(memory)
        
        # Log counts by source type
        source_counts = {}
        for mem in filtered_memories:
            source = mem.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        # print(f"🆕 Initialized store '{trial_namespace}' with {len(filtered_memories)} memories: {source_counts}")
        
        # Populate store with filtered memories using agent-friendly IDs
        for memory in filtered_memories:
            content = memory.get('content', '')
            # Extract metadata from the memory structure
            # Initial memories have fields like memory_type, speaker, evidence, etc.
            metadata = {
                "type": memory.get('memory_type', 'episodic'),  # All LoCoMo data are episodic
                "speaker": memory.get('speaker', ''),
                "evidence": memory.get('evidence', ''),
                "session": memory.get('session', ''),
                "timestamp": memory.get('timestamp', 'unknown time'),  # Add timestamp field
                "source": memory.get('source', 'observation')  # Track data level source
            }
            
            # Generate sequential ID for initial memory
            memory_id = self._generate_agent_friendly_id(trial_namespace)
            
            try:
                await store.aput(
                    ("memories",),
                    memory_id,
                    {
                        "content": content,
                        "metadata": metadata,
                        "id": memory_id
                    }
                )
            except Exception as e:
                print(f"❌ Failed to add initial memory {memory_id}: {e}")
                
        # Log the loaded memory IDs for debugging
        initial_ids = []
        for mem in await store.asearch(("memories",), query="", limit=5):
            initial_ids.append(mem.key)

        # zycyc debug prints
        # print(f"✅ [MEMORY_INIT] Loaded {len(filtered_memories)} memories for {trial_namespace}: {initial_ids[:3]}...")

        return trial_namespace

    async def get_current_memory(self, trial_namespace: str) -> List[Dict]:
        """Get final memory state for reward computation (called on reward workers)."""
        if trial_namespace not in self._stores:
            print(f"⚠️  Namespace '{trial_namespace}' not found in get_current_memory")
            # Return empty list if trial_namespace was never created yet
            return []
        
        store = self._stores[trial_namespace]
        
        try:
            # Query all memories from the actual LangMem store
            memories = await store.asearch(
                ("memories",),  # namespace_prefix as positional argument
                query="",  # Empty query returns all memories
                limit=999999  # High limit to get all memories
            )
        
            
            # Convert to reward manager format
            result = []
            for mem in memories:
                result.append({
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "metadata": mem.value.get("metadata", {}),
                    "trial_namespace": trial_namespace,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "updated_at": mem.updated_at.isoformat() if mem.updated_at else None,
                })
            
            return result
            
        except Exception as e:
            print(f"❌ CRITICAL: Error querying memories for {trial_namespace}: {type(e).__name__}: {e}")
    
    async def cleanup_conversation(self, trial_namespace: str):
        """Clean up memory for a conversation after episode completion."""
        # print(f"🧹 In cleanup_conversation, trial_namespace {trial_namespace} is in stores: {trial_namespace in self._stores}")
        if trial_namespace in self._stores:
            # Simple cleanup - the original approach
            del self._stores[trial_namespace]
            
            # Force garbage collection to help with memory cleanup
            import gc
            gc.collect()


class MemoryStoreManager:
    """
    Updated MemoryStoreManager that uses Ray Actor for cross-process memory sharing.
    
    This class is instantiated on both:
    - Rollout workers (for tool execution)
    - Reward workers (for reward computation)
    """
    
    _broker_actor: Optional[ray.actor.ActorHandle] = None
    _broker_actors: Dict[int, ray.actor.ActorHandle] = {}  # Sharded actors
    _num_shards: int = 32  # Default number of memory broker shards (configurable)
    _use_sharding: bool = True  # Enable sharding by default
    _instance_to_namespace: Dict[str, str] = {}

    @classmethod
    def configure_sharding(cls, num_shards: int):
        """Configure the number of memory broker shards."""
        cls._num_shards = num_shards
        print(f"🔧 Configured MemoryStoreManager with {num_shards} shards")
    
    @classmethod
    def _get_shard_id(cls, trial_namespace: str) -> int:
        """Get deterministic shard ID for trial_namespace using random distribution."""
        if not cls._use_sharding:
            return 0
        
        # Simple solution: Just count how many memory broker actors actually exist
        if cls._num_shards == 32:  # Still default, check what's actually available
            actual_shard_count = 0
            for i in range(128):  # Check reasonable range
                try:
                    ray.get_actor(f"memory_broker_{i}")
                    actual_shard_count += 1
                except ValueError:
                    break
            if actual_shard_count > 0:
                cls._num_shards = actual_shard_count
            
        # Use full trial_namespace for random distribution (better load balancing)
        # This distributes similar conversations across different shards
        import hashlib
        hash_obj = hashlib.md5(trial_namespace.encode())
        shard_id = int(hash_obj.hexdigest(), 16) % cls._num_shards
        
        return shard_id
    
    @classmethod
    def initialize_shard_cache(cls):
        """Pre-cache all shard actors to avoid race conditions."""
        if not cls._use_sharding:
            return
            
        print(f"🔄 Pre-caching {cls._num_shards} memory broker shards...")
        for shard_id in range(cls._num_shards):
            try:
                actor_name = f"memory_broker_{shard_id}"
                cls._broker_actors[shard_id] = ray.get_actor(actor_name)
            except ValueError:
                print(f"⚠️ Could not find shard {actor_name}")
        print(f"✅ Cached {len(cls._broker_actors)} memory broker shards")
    
    @classmethod
    def get_broker_actor(cls, trial_namespace: str = None) -> ray.actor.ActorHandle:
        """Get the appropriate memory broker actor for trial_namespace."""
        # Try sharded actors first
        if cls._use_sharding and trial_namespace:  # Removed len(cls._broker_actors) > 0 check
            shard_id = cls._get_shard_id(trial_namespace)
            if shard_id in cls._broker_actors:
                return cls._broker_actors[shard_id]
            
            # Load sharded actor if not cached
            try:
                actor_name = f"memory_broker_{shard_id}"
                cls._broker_actors[shard_id] = ray.get_actor(actor_name)
                # Log shard routing for debugging
                # print(f"📍 Routing namespace '{trial_namespace[:30]}...' to {actor_name}")
                return cls._broker_actors[shard_id]
            except ValueError:
                print(f"⚠️ Sharded actor {actor_name} not found, falling back to single actor")
                cls._use_sharding = False
        
        # Fallback to single actor (backward compatibility)
        if cls._broker_actor is None:
            try:
                cls._broker_actor = ray.get_actor("memory_broker")
                print(f"📡 Connected to fallback MemoryBrokerActor from process {os.getpid()}")
            except ValueError:
                raise RuntimeError("MemoryBrokerActor 'memory_broker' not found. It should be initialized before training starts.")
                        
        return cls._broker_actor
    
    @classmethod
    def get_initial_memories(cls, sample_id: str) -> List[Dict]:
        """Get initial memories for a sample_id from the broker."""
        # Use sample_id as trial_namespace for routing (same conversation)
        broker = cls.get_broker_actor(sample_id)
        return ray.get(broker.get_initial_memories.remote(sample_id))
    
    @classmethod
    def get_conversation_stats(cls) -> Dict[str, Any]:
        """Get conversation statistics from all brokers."""
        # Always try sharded actors first if available
        try:
            # Try to get stats from shard 0 as representative
            broker = ray.get_actor("memory_broker_0")
            return ray.get(broker.get_conversation_stats.remote())
        except ValueError:
            # Fallback to single actor
            try:
                broker = ray.get_actor("memory_broker")
                return ray.get(broker.get_conversation_stats.remote())
            except ValueError:
                return {"error": "No memory broker actors found"}

    @classmethod
    def get_current_memory(cls, trial_namespace: str) -> List[Dict]:
        """Get current memory state for reward computation (called on reward workers)."""
        broker = cls.get_broker_actor(trial_namespace)
        return ray.get(broker.get_current_memory.remote(trial_namespace))
    
    @classmethod
    async def get_current_memory_async(cls, trial_namespace: str) -> List[Dict]:
        """Async version for tool contexts to avoid blocking event loop."""
        import time
        start_time = time.perf_counter()
        
        broker = cls.get_broker_actor(trial_namespace)
        result = await broker.get_current_memory.remote(trial_namespace)
        
        end_time = time.perf_counter()
        
        # Track timing metrics
        if not hasattr(cls, '_memory_timing_stats'):
            cls._memory_timing_stats = {}
        if 'get_current_memory' not in cls._memory_timing_stats:
            cls._memory_timing_stats['get_current_memory'] = []
        cls._memory_timing_stats['get_current_memory'].append(end_time - start_time)
        
        return result
    
    @classmethod
    async def create_memory_via_actor(cls, trial_namespace: str, memory_data: dict) -> dict:
        """Create memory directly in Ray Actor store (fixes serialization issue)."""
        import time
        start_time = time.perf_counter()
        
        # print(f"➕ Creating memory in namespace '{trial_namespace}' via actor from process {os.getpid()}")
        broker = cls.get_broker_actor(trial_namespace)
        result = await broker.create_memory_in_store.remote(trial_namespace, memory_data)
        
        end_time = time.perf_counter()
        
        # Track timing metrics
        if not hasattr(cls, '_memory_timing_stats'):
            cls._memory_timing_stats = {}
        if 'create_memory' not in cls._memory_timing_stats:
            cls._memory_timing_stats['create_memory'] = []
        cls._memory_timing_stats['create_memory'].append(end_time - start_time)
        
        return result
    
    @classmethod
    async def update_memory_via_actor(cls, trial_namespace: str, memory_data: dict) -> dict:
        """Update memory directly in Ray Actor store (fixes serialization issue)."""
        import time
        start_time = time.perf_counter()
        
        # print(f"📝 Updating memory in namespace '{trial_namespace}' via actor from process {os.getpid()}")
        broker = cls.get_broker_actor(trial_namespace)
        result = await broker.update_memory_in_store.remote(trial_namespace, memory_data)
        
        end_time = time.perf_counter()
        
        # Track timing metrics
        if not hasattr(cls, '_memory_timing_stats'):
            cls._memory_timing_stats = {}
        if 'update_memory' not in cls._memory_timing_stats:
            cls._memory_timing_stats['update_memory'] = []
        cls._memory_timing_stats['update_memory'].append(end_time - start_time)
        
        return result
    
    @classmethod
    def search_memory_via_actor(cls, trial_namespace: str, query: str, limit: int = 5) -> dict:
        """Search memory directly in Ray Actor store (fixes serialization issue)."""
        import time
        start_time = time.perf_counter()
        
        broker = cls.get_broker_actor(trial_namespace)
        result = ray.get(broker.search_memory_in_store.remote(trial_namespace, query, limit))
        
        end_time = time.perf_counter()
        
        # Track timing metrics
        if not hasattr(cls, '_memory_timing_stats'):
            cls._memory_timing_stats = {}
        if 'search_memory' not in cls._memory_timing_stats:
            cls._memory_timing_stats['search_memory'] = []
        cls._memory_timing_stats['search_memory'].append(end_time - start_time)
        
        return result
    
    @classmethod
    async def search_memory_via_actor_async(cls, trial_namespace: str, query: str, limit: Optional[int] = 5, source_filter: str = "", speaker_filter: str = "", session_filter: str = "", list_all: bool = False) -> dict:
        """Async version for tool execution contexts with multi-field filtering."""
        import time
        start_time = time.perf_counter()
        
        broker = cls.get_broker_actor(trial_namespace)
        result = await broker.search_memory_in_store.remote(trial_namespace, query, limit, source_filter, speaker_filter, session_filter, list_all)
        
        end_time = time.perf_counter()
        
        # Track timing metrics
        if not hasattr(cls, '_memory_timing_stats'):
            cls._memory_timing_stats = {}
        if 'search_memory_async' not in cls._memory_timing_stats:
            cls._memory_timing_stats['search_memory_async'] = []
        cls._memory_timing_stats['search_memory_async'].append(end_time - start_time)
        
        return result

    @classmethod
    def register_instance_namespace(cls, instance_id: str, trial_namespace: str):
        """Register mapping from instance_id (request_id) to conversation trial_namespace."""
        cls._instance_to_namespace[instance_id] = trial_namespace
    
    @classmethod
    def get_namespace_for_instance(cls, instance_id: str) -> str:
        """Get trial_namespace for instance_id (request_id)."""
        trial_namespace = cls._instance_to_namespace.get(instance_id, instance_id)
        return trial_namespace
    
    @classmethod
    def init_conversation_memory(cls, trial_namespace: str, sample_id: str,
                                include_observation: bool = True,
                                include_conversation: bool = True,
                                include_event_summary: bool = True) -> bool:
        """Ensure store is initialized, but only if it doesn't exist (idempotent)."""
        broker = cls.get_broker_actor(trial_namespace)
        return ray.get(broker.init_conversation_memory.remote(
            trial_namespace, sample_id, 
            include_observation, include_conversation, include_event_summary
        ))
    
    @classmethod
    def cleanup_conversation(cls, trial_namespace: str):
        """Clean up memory after episode completion."""
        broker = cls.get_broker_actor(trial_namespace)
        ray.get(broker.cleanup_conversation.remote(trial_namespace))
    
    @classmethod
    async def cleanup_conversation_async(cls, trial_namespace: str):
        """Clean up memory after episode completion (async version)."""
        broker = cls.get_broker_actor(trial_namespace)
        await broker.cleanup_conversation.remote(trial_namespace)
    
    @classmethod
    def register_rollout_embedding_model(cls, worker_id: str):
        """MEMUPDATE: Register rollout worker ID for direct embedding access."""
        # Register with all shards for load balancing
        if cls._use_sharding:
            for shard_id in range(cls._num_shards):
                try:
                    # Directly get the actor by name to avoid routing issues
                    actor_name = f"memory_broker_{shard_id}"
                    broker = ray.get_actor(actor_name)
                    ray.get(broker.register_embedding_model.remote(worker_id))
                except:
                    continue
        else:
            # Use a dummy namespace to ensure proper routing even without sharding
            broker = cls.get_broker_actor(f"worker_{worker_id}")
            ray.get(broker.register_embedding_model.remote(worker_id))


class MockMemoryStore:
    """Mock store for testing without LangMem."""

    def __init__(self, trial_namespace: str):
        self.trial_namespace = trial_namespace
        self.memories = []
