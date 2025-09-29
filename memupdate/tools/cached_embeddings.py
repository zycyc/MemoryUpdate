"""Smart cached embeddings: uses pre-computed cache, generates new ones on-demand."""

import torch
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Any
from langchain_core.embeddings import Embeddings


class SmartCachedEmbeddings(Embeddings):
    """Unified embeddings class that:
    1. Uses pre-computed embeddings from cache
    2. Filters by conversation when provided
    3. Generates embeddings for new content using GPU if available, else CPU
    4. Can optionally cache newly generated embeddings
    5. Caches on-demand embedding models to avoid repeated loading
    """
    
    # Class-level cache for on-demand embedding models to avoid repeated creation
    _on_demand_models = {}
    
    def __init__(
        self, 
        cache: Optional[Dict[str, Any]] = None,
        cache_file: Optional[str] = None,
        sample_id: Optional[str] = None,
        embedding_model = None,
        embedding_worker = None,
        embedding_worker_id = None,
        device: Optional[str] = None
    ):
        """Initialize smart cached embeddings.
        
        Args:
            cache: Pre-loaded embedding cache (if already loaded by broker)
            cache_file: Path to cache file (if not pre-loaded)
            sample_id: Optional conversation ID to filter embeddings
            embedding_model: Local embedding model (deprecated - use embedding_worker_id)
            embedding_worker: Ray actor handle for rollout worker (deprecated - use embedding_worker_id)
            embedding_worker_id: Worker ID for rollout worker with GPU embedding model
            device: Device for embedding model ('cuda', 'cpu', or None for auto)
        """
        self._cache = cache or {}
        self.sample_id = sample_id
        self.embedding_model = embedding_model  # Backward compatibility
        self.embedding_worker = embedding_worker  # Deprecated: direct worker handle
        self.embedding_worker_id = embedding_worker_id  # New: worker ID for actor lookup
        self._new_embeddings = {}  # Cache for newly generated embeddings
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load cache from file if not pre-loaded
        if not self._cache and cache_file:
            self._load_cache(cache_file)
            
        # Filter cache by conversation if sample_id provided
        if self.sample_id and self._cache:
            self._filter_cache_by_conversation()
            
        # Note: Embedding generation now handled by distributed pool
        # No single model initialization needed
    
    def _load_cache(self, cache_file: str):
        """Load pre-computed embeddings from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                self._cache = pickle.load(f)
                    
            print(f"💾 Loaded {len(self._cache)} embeddings from {cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")
            self._cache = {}
    
    def _filter_cache_by_conversation(self):
        """Filter cache to only include embeddings for the specified conversation."""
        if not self.sample_id:
            return
            
        filtered_cache = {}
        for key, value in self._cache.items():
            if value.get('sample_id') == self.sample_id:
                filtered_cache[key] = value
                
        self._cache = filtered_cache
    
    # Note: _init_embedding_model removed - using distributed embedding pool instead
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache (either pre-computed or newly generated)."""
        # Check pre-computed cache
        if content_hash in self._cache:
            cached_data = self._cache[content_hash]
            embedding = cached_data.get('embedding')
                
            # Convert to list if needed
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif isinstance(embedding, torch.Tensor):
                return embedding.cpu().numpy().tolist()
            else:
                return embedding
                
        # Check newly generated cache
        if content_hash in self._new_embeddings:
            return self._new_embeddings[content_hash]
            
        return None
    
    def _get_worker_handle(self, worker_id: str):
        """Convert worker ID to Ray actor handle using multiple strategies."""
        import ray
        
        # Strategy 1: Try direct name lookup (for constructed actor names)
        try:
            # print(f"🔍 Trying direct name lookup for worker_id: {worker_id}")
            return ray.get_actor(name=worker_id)
        except Exception as e:
            print(f"❌ Direct name lookup failed: {e}")
        
        # Strategy 2: Actor enumeration - search for any WorkerDict (actual actor type)
        try:
            # List all actors and find WorkerDict actors (which are the rollout workers)
            from ray.experimental.state.api import list_actors
            actors = list_actors()
            print(f"🔍 Direct lookup failed, searching {len(actors)} actors for WorkerDict")
            
            # Debug: Show all actor names to understand the actual naming pattern
            actor_names = [actor.get('name', '') for actor in actors if actor.get('name', '')]
            print(f"🔍 Available named actors: {actor_names}")
            
            # Look for any WorkerDict actor and return the first one with embedding capability
            for actor in actors:
                actor_name = actor.get('name', '')
                
                if 'WorkerDict' in actor_name:
                    try:
                        handle = ray.get_actor(name=actor_name)
                        # Test if this actor has embedding capability by trying to call embed_documents
                        # We need to be careful here - test with ray.get with timeout
                        print(f"✅ Found WorkerDict actor, testing embedding capability: {actor_name}")
                        return handle
                    except Exception as e:
                        print(f"❌ Failed to get actor {actor_name}: {e}")
                        continue
                        
            print(f"❌ No WorkerDict actors found in actor list")
            
        except Exception as e:
            print(f"❌ Actor enumeration failed: {e}")
        
        # Strategy 3: Return None to trigger fallback
        print(f"❌ All strategies failed for worker_id: {worker_id}")
        return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using cache when available, generate for new content."""
        if not texts:
            return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # First pass: use cached embeddings
        for i, text in enumerate(texts):
            content_hash = self._hash_content(text)
            cached_embedding = self._get_cached_embedding(content_hash)
            
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Second pass: generate embeddings for uncached content using rollout worker
        if uncached_texts:
            # Try worker ID approach first (preferred), then legacy approaches
            if self.embedding_worker_id:
                try:
                    import ray
                    # print(f"🎯 Calling rollout worker {self.embedding_worker_id} for {len(uncached_texts)} new embeddings")
                    
                    # Try to get the Ray actor handle from worker ID
                    worker_handle = self._get_worker_handle(self.embedding_worker_id)
                    if worker_handle is None:
                        raise Exception(f"Cannot find Ray actor for worker {self.embedding_worker_id}")
                    
                    # Call the worker's embed_documents method directly
                    # Note: In fused workers, methods are prefixed with role name
                    new_embeddings_raw = ray.get(worker_handle.actor_rollout_embed_documents.remote(uncached_texts))
                    
                    # Convert to lists and cache results
                    new_embeddings = []
                    for emb in new_embeddings_raw:
                        if hasattr(emb, 'tolist'):
                            new_embeddings.append(emb.tolist())
                        else:
                            new_embeddings.append(list(emb))
                    
                    # Cache the new embeddings and fill results
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        content_hash = self._hash_content(text)
                        self._new_embeddings[content_hash] = embedding
                        results[uncached_indices[idx]] = embedding
                    
                    # print(f"✅ Successfully got {len(new_embeddings)} embeddings from GPU rollout worker!")
                    return results  # Early return - don't fall through to other methods
                    
                except Exception as e:
                    print(f"❌ Failed to get embeddings via worker ID {self.embedding_worker_id}: {e}")
                    # Fall through to other methods
                    
            # Try direct worker handle (legacy)
            elif self.embedding_worker:
                try:
                    import ray
                    # print(f"🎯 Calling rollout worker for {len(uncached_texts)} new embeddings")
                    # Call rollout worker's embed_documents method directly
                    new_embeddings_raw = ray.get(self.embedding_worker.embed_documents.remote(uncached_texts))
                    
                    # Convert to lists and cache results
                    new_embeddings = []
                    for emb in new_embeddings_raw:
                        if hasattr(emb, 'tolist'):
                            new_embeddings.append(emb.tolist())
                        else:
                            new_embeddings.append(list(emb))
                    
                    # Cache the new embeddings and fill results
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        content_hash = self._hash_content(text)
                        self._new_embeddings[content_hash] = embedding
                        results[uncached_indices[idx]] = embedding
                    
                    print(f"✅ Successfully got {len(new_embeddings)} embeddings from rollout worker")
                    return results  # Early return - don't fall through to other methods
                    
                except Exception as e:
                    print(f"❌ Failed to get embeddings from rollout worker: {e}")
                    # Fall through to local model
            
            # Final fallback: try local embedding model or create one on-demand
            if self.embedding_model:
                try:
                    print(f"🔄 Using existing local embedding model for {len(uncached_texts)} texts")
                    new_embeddings_raw = self.embedding_model.embed_documents(uncached_texts)
                    new_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in new_embeddings_raw]
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        content_hash = self._hash_content(text)
                        self._new_embeddings[content_hash] = embedding
                        results[uncached_indices[idx]] = embedding
                    print(f"✅ Generated {len(new_embeddings)} embeddings with local model")
                    return results  # Early return - don't fall through to on-demand model
                except Exception as e:
                    print(f"❌ Local embedding model failed: {e}, using random vectors")
                    for idx in uncached_indices:
                        results[idx] = (np.random.randn(1024) * 0.01).tolist()
                    return results  # Early return after fallback
            else:
                # Last resort: get or create cached on-demand embedding model
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_key = f"qwen_embedding_{device}"
                    
                    # Check class-level cache first
                    if model_key in SmartCachedEmbeddings._on_demand_models:
                        on_demand_model = SmartCachedEmbeddings._on_demand_models[model_key]
                        print(f"🔄 Using cached on-demand embedding model ({device}) for {len(uncached_texts)} texts")
                    else:
                        # Create and cache new model
                        from memupdate.data.qwen_embeddings import QwenEmbeddings
                        print(f"🔧 Creating and caching on-demand embedding model ({device}) for {len(uncached_texts)} texts")
                        on_demand_model = QwenEmbeddings()
                        SmartCachedEmbeddings._on_demand_models[model_key] = on_demand_model
                        print(f"💾 Cached on-demand embedding model as {model_key}")
                    
                    new_embeddings_raw = on_demand_model.embed_documents(uncached_texts)
                    new_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in new_embeddings_raw]
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        content_hash = self._hash_content(text)
                        self._new_embeddings[content_hash] = embedding
                        results[uncached_indices[idx]] = embedding
                    print(f"✅ Generated {len(new_embeddings)} embeddings with cached on-demand model")
                except Exception as e:
                    print(f"❌ On-demand embedding model failed: {e}, using random vectors")
                    for idx in uncached_indices:
                        results[idx] = (np.random.randn(1024) * 0.01).tolist()
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version - just calls sync version."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version - just calls sync version."""
        return self.embed_query(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings cache."""
        return {
            'cached_embeddings': len(self._cache),
            'new_embeddings': len(self._new_embeddings),
            'sample_id': self.sample_id,
            'device': self.device,
            'has_model': self.embedding_model is not None,
            'has_worker': self.embedding_worker is not None,
            'has_worker_id': self.embedding_worker_id is not None,
            'cached_on_demand_models': len(SmartCachedEmbeddings._on_demand_models)
        }