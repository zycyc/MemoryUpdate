#!/usr/bin/env python3
"""Pre-compute embeddings for all memory content in LoCoMo training data.

This script:
1. Loads the Qwen3-Embedding-0.6B model
2. Extracts all unique memory content from training data
3. Computes embeddings for each unique content
4. Saves to memory_embeddings.pkl for fast lookup during training

Usage:
    python3 precompute_embeddings.py --input /workspace/locomo/data/locomo10.json
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np

# Add memupdate to path
sys.path.append('/workspace/memupdate')

def hash_content(content: str) -> str:
    """Generate hash for content (matches cached_embeddings.py)."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def extract_memories_from_json(file_path: str, 
                              include_observation: bool = True,
                              include_conversation: bool = True,
                              include_event_summary: bool = True) -> Dict[str, tuple]:
    """Extract unique memory content from LoCoMo JSON file with all data levels.
    
    Returns:
        Dict mapping content -> (sample_id, session, speaker, source, timestamp)
    """
    print(f"📖 Loading memories from JSON: {file_path}")
    print(f"  Data levels: observation={include_observation}, conversation={include_conversation}, event_summary={include_event_summary}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} conversations from JSON")
        
        # Map content to its metadata (sample_id, session, speaker, source, timestamp)
        memory_map = {}
        
        for conversation in data:
            sample_id = conversation.get('sample_id', 'unknown')
            
            # Extract session timestamps from conversation level
            timestamps = {}
            if 'conversation' in conversation:
                for key in conversation['conversation'].keys():
                    if key.endswith('_date_time'):
                        session_num = key.replace('_date_time', '')
                        timestamps[session_num] = conversation['conversation'][key]
            
            # 1. Process observations (extracted facts)
            if include_observation and 'observation' in conversation:
                observations = conversation['observation']
                for session_key, session_obs in observations.items():
                    if isinstance(session_obs, dict):
                        session_num = session_key.replace('_observation', '')
                        timestamp = timestamps.get(session_num, 'unknown time')
                        for speaker, speaker_observation in session_obs.items():
                            for fact_entry in speaker_observation:
                                if isinstance(fact_entry, list) and len(fact_entry) >= 2:
                                    fact_text, evidence = fact_entry[0], fact_entry[1]
                                    content = fact_text.strip()
                                    if content:
                                        # Store with extended metadata
                                        memory_map[content] = (sample_id, session_num, speaker, 'observation', timestamp)
            
            # 2. Process conversation dialogue
            if include_conversation and 'conversation' in conversation:
                conv_data = conversation['conversation']
                for session_key, dialogue_data in conv_data.items():
                    if session_key.startswith('session_') and not session_key.endswith('_date_time'):
                        timestamp = timestamps.get(session_key, 'unknown time')
                        if isinstance(dialogue_data, list):
                            for turn in dialogue_data:
                                if isinstance(turn, dict) and 'text' in turn:
                                    content = turn.get('text', '').strip()
                                    if content:
                                        speaker = turn.get('speaker', '')
                                        memory_map[content] = (sample_id, session_key, speaker, 'conversation', timestamp)
            
            # 3. Process event summaries
            if include_event_summary and 'event_summary' in conversation:
                events = conversation['event_summary']
                for event_key, event_data in events.items():
                    if isinstance(event_data, dict):
                        session_num = event_key.replace('events_', '')
                        timestamp = timestamps.get(session_num, event_data.get('date', 'unknown time'))
                        for speaker, event_list in event_data.items():
                            if speaker != 'date' and isinstance(event_list, list):
                                for event in event_list:
                                    if event and isinstance(event, str):
                                        content = event.strip()
                                        if content:
                                            memory_map[content] = (sample_id, session_num, speaker, 'event_summary', timestamp)
        
        # Count by source type
        source_counts = {}
        for _, metadata in memory_map.items():
            source = metadata[3] if len(metadata) > 3 else 'unknown'
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"✅ Extracted {len(memory_map)} unique memory contents")
        print(f"   By source: {source_counts}")
        return memory_map
        
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        return {}

def compute_embeddings_batch(embedding_model, memory_map: Dict[str, tuple], batch_size: int = 32) -> Dict[str, dict]:
    """Compute embeddings for a batch of contents with metadata.
    
    Args:
        embedding_model: Model to compute embeddings
        memory_map: Dict mapping content -> (sample_id, session, speaker)
        batch_size: Batch size for processing
        
    Returns:
        Dict mapping hash -> {embedding, sample_id, content}
    """
    contents = list(memory_map.keys())
    print(f"🔄 Computing embeddings for {len(contents)} contents (batch size: {batch_size})")
    
    cache = {}
    
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        
        try:
            # Compute embeddings for batch
            batch_embeddings = embedding_model.embed_documents(batch_contents)
            
            # Store in cache with hash keys and extended metadata
            for content, embedding in zip(batch_contents, batch_embeddings):
                content_hash = hash_content(content)
                metadata = memory_map[content]
                
                # Handle both old format (3 items) and new format (5 items)
                if len(metadata) >= 5:
                    sample_id, session, speaker, source, timestamp = metadata
                else:
                    # Fallback for old format
                    sample_id, session, speaker = metadata
                    source = 'observation'
                    timestamp = 'unknown time'
                
                cache[content_hash] = {
                    'embedding': np.array(embedding, dtype=np.float32),
                    'sample_id': sample_id,
                    'content': content,
                    'session': session,
                    'speaker': speaker,
                    'source': source,
                    'timestamp': timestamp
                }
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
            # Skip this batch and continue
            continue
    
    return cache

def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for memory content")
    parser.add_argument(
        "--input",
        default="/workspace/memupdate/data/locomo10.json",
        help="Input file (JSON). Defaults to LoCoMo dataset JSON.",
    )
    parser.add_argument("--output", default="/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl", 
                       help="Output pickle file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--dry-run", action="store_true", help="Just count unique memories, don't compute embeddings")
    parser.add_argument("--include-observation", action="store_true", default=True, 
                       help="Include observation-level memories (default: True)")
    parser.add_argument("--include-conversation", action="store_true", default=True,
                       help="Include conversation-level dialogue (default: True)")
    parser.add_argument("--include-event-summary", action="store_true", default=True,
                       help="Include event summary memories (default: True)")
    parser.add_argument("--no-observation", dest="include_observation", action="store_false",
                       help="Exclude observation-level memories")
    parser.add_argument("--no-conversation", dest="include_conversation", action="store_false",
                       help="Exclude conversation-level dialogue")
    parser.add_argument("--no-event-summary", dest="include_event_summary", action="store_false",
                       help="Exclude event summary memories")
    
    args = parser.parse_args()
    
    print("🚀 Starting embedding pre-computation...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    # Extract unique memory contents with metadata
    if args.input.endswith('.json'):
        memory_map = extract_memories_from_json(
            args.input,
            include_observation=args.include_observation,
            include_conversation=args.include_conversation,
            include_event_summary=args.include_event_summary
        )
    else:
        print(f"❌ Unsupported file format: {args.input}")
        return
    
    if not memory_map:
        print("❌ No memories found in input file")
        return
    
    print(f"📊 Found {len(memory_map)} unique memory contents")
    
    # Show conversation and source distribution
    conv_counts = {}
    source_counts = {}
    for content, metadata in memory_map.items():
        sample_id = metadata[0]
        conv_counts[sample_id] = conv_counts.get(sample_id, 0) + 1
        # Handle both old and new format
        source = metadata[3] if len(metadata) > 3 else 'observation'
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"📊 Source distribution: {source_counts}")
    print(f"📚 Memory distribution across {len(conv_counts)} conversations:")
    for conv_id, count in sorted(conv_counts.items())[:5]:
        print(f"  {conv_id}: {count} memories")
    
    # Show some examples
    memory_list = list(memory_map.keys())[:5]
    print("📝 Sample memories:")
    for i, memory in enumerate(memory_list):
        sample_id = memory_map[memory][0]
        print(f"  {i+1}. [{sample_id}] {memory[:80]}{'...' if len(memory) > 80 else ''}")
    
    # Load embedding model
    try:
        # Use the in-repo embedding wrapper that calls the embedding model directly
        from memupdate.data.qwen_embeddings import QwenEmbeddings
        embedding_model = QwenEmbeddings()
        print("✅ Loaded QwenEmbeddings successfully")
    except Exception as e:
        print(f"❌ Failed to load QwenEmbeddings: {e}")
        return
    
    # Compute embeddings with metadata
    embedding_cache = compute_embeddings_batch(embedding_model, memory_map, args.batch_size)
    
    print(f"💾 Saving {len(embedding_cache)} embeddings to {args.output}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(embedding_cache, f)
    
    # Verify the saved file
    try:
        with open(args.output, 'rb') as f:
            loaded_cache = pickle.load(f)
        
        print(f"✅ Verification: Loaded {len(loaded_cache)} embeddings from saved file")
        
        # Show sample with metadata
        sample_key = list(loaded_cache.keys())[0]
        sample_data = loaded_cache[sample_key]
        
        print(f"📊 Sample embedding shape: {sample_data['embedding'].shape}, dtype: {sample_data['embedding'].dtype}")
        print(f"   Sample ID: {sample_data.get('sample_id', 'N/A')}")
        print(f"   Content preview: {sample_data.get('content', 'N/A')[:50]}...")
        
    except Exception as e:
        print(f"❌ Failed to verify saved file: {e}")
    
    print("🎉 Embedding pre-computation complete!")

if __name__ == "__main__":
    main()
