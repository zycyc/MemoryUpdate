"""LoCoMo dataset preprocessing for verl training format."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger(__name__)


class LoCoMoProcessor:
    """Processes LoCoMo dataset for MemUpdate training."""

    def __init__(self, locomo_data_path: str = "/workspace/memupdate/data/locomo10.json",
                 include_observation: bool = True,
                 include_conversation: bool = True,
                 include_event_summary: bool = True):
        self.locomo_data_path = Path(locomo_data_path)
        self.data = None
        self.include_observation = include_observation
        self.include_conversation = include_conversation
        self.include_event_summary = include_event_summary

    def load_data(self):
        """Load LoCoMo dataset from JSON file."""
        try:
            with open(self.locomo_data_path, "r") as f:
                self.data = json.load(f)
            logger.info(
                f"Loaded {len(self.data)} conversations from {self.locomo_data_path}"
            )
        except FileNotFoundError:
            logger.error(f"LoCoMo data file not found: {self.locomo_data_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LoCoMo JSON data: {e}")
            raise

    def create_train_test_split(self, train_conversations: int = 7, seed: int = 42):
        """Create train/test split from conversations."""
        if not self.data:
            self.load_data()

        random.seed(seed)
        conversations = list(range(len(self.data)))
        random.shuffle(conversations)

        train_indices = conversations[:train_conversations]
        test_indices = conversations[train_conversations:]

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]

        logger.info(
            f"Split data: {len(train_data)} train conversations, {len(test_data)} test conversations"
        )
        return train_data, test_data

    def extract_qa_pairs(self, conversations: List[Dict]) -> List[Dict]:
        """Extract QA pairs from conversations for training trials."""
        qa_trials = []
        import uuid

        for conv in conversations:
            sample_id = conv.get("sample_id", "unknown") # like conv-48
            qa_pairs = conv.get("qa", [])

            for qa_idx, qa in enumerate(qa_pairs):
                if qa['category'] == 5: # adversirial category, not used in Mem0 so pass for now:
                    continue
                # 🔧 CRITICAL FIX: Create unique trajectory ID for each QA pair
                # Each QA pair gets its own namespace to prevent memory bank sharing
                trajectory_id = f"{sample_id}-qa{qa_idx}-{str(uuid.uuid4())[:8]}"
                
                trial = {
                    "sample_id": sample_id,  # Original conversation ID
                    "trajectory_id": trajectory_id,  # Unique trajectory ID
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category", 0),
                }
                qa_trials.append(trial)

        logger.info(f"Extracted {len(qa_trials)} QA trials with unique trajectory IDs")
        return qa_trials

    def get_memory_context_description(self, sample_id: str) -> str:
        """Generate descriptive context about the initial memory database for a conversation."""
        if not self.data:
            self.load_data()
        
        # Find the conversation by sample_id
        conversation = None
        for conv in self.data:
            if conv.get("sample_id") == sample_id:
                conversation = conv
                break
        
        if not conversation:
            return "Memory database information unavailable."
        
        # Count memories by type and session
        observation_count = 0
        conversation_count = 0
        event_summary_count = 0
        speakers = set()
        num_sessions = 0
        
        # Count observation memories
        if self.include_observation and "observation" in conversation:
            observations = conversation.get("observation", {})
            for session_key, session_obs in observations.items():
                if isinstance(session_obs, dict):
                    for speaker, speaker_observation in session_obs.items():
                        speakers.add(speaker)
                        observation_count += len(speaker_observation)
        
        # Count conversation memories (dialogue turns)
        if self.include_conversation and "conversation" in conversation:
            conv_data = conversation.get("conversation", {})
            for session_key, dialogue_list in conv_data.items():
                if session_key.startswith('session_') and not session_key.endswith('_date_time'):
                    num_sessions = max(num_sessions, int(session_key.split('_')[1]) if session_key.split('_')[1].isdigit() else 0)
                    if isinstance(dialogue_list, list):
                        for turn in dialogue_list:
                            if isinstance(turn, dict) and 'speaker' in turn:
                                speakers.add(turn['speaker'])
                                conversation_count += 1
        
        # Count event summary memories
        if self.include_event_summary and "event_summary" in conversation:
            events = conversation.get("event_summary", {})
            for event_key, event_data in events.items():
                if isinstance(event_data, dict):
                    for speaker, event_list in event_data.items():
                        if speaker != 'date' and isinstance(event_list, list):
                            if speaker:
                                speakers.add(speaker)
                            event_summary_count += len(event_list)
        
        # Generate descriptive context
        speakers_list = sorted(list(speakers))
        total_memories = observation_count + conversation_count + event_summary_count
        
        # Build context description based on enabled levels
        memory_breakdown = []
        if self.include_conversation:
            memory_breakdown.append(f"{conversation_count} conversation dialogue turns")
        if self.include_observation:
            memory_breakdown.append(f"{observation_count} extracted observations")
        if self.include_event_summary:
            memory_breakdown.append(f"{event_summary_count} event summaries")
        
        breakdown_str = ", ".join(memory_breakdown) if memory_breakdown else "no memories"
        
        context = f"""You have access to the following memory database: {total_memories} total memories ({breakdown_str}) across {num_sessions} sessions between {' and '.join(speakers_list)}.
The database contains three levels of information: raw conversation dialogue, extracted observations, and event summaries.
Each memory includes speaker, session, timestamp, and source metadata."""
        
        return context

    def create_verl_training_data(self, qa_trials: List[Dict]) -> List[Dict]:
        """Convert QA trials to verl training format with full context in messages."""
        training_data = []

        for idx, trial in enumerate(qa_trials):
            # Create system prompt (tools will be auto-injected by tokenizer)
            system_content = """You are an expert at exploiting memory databases for question-answering. Your goal is to search over an existing memory database using tools, reason through them, and create new memories for answering a target question. Available tools are described below.
"""

            # Get memory context description for this conversation
            memory_context = self.get_memory_context_description(trial["sample_id"])
            
            # Create user prompt with memory context
            user_content = f"""{memory_context}

**Your task is to exploit the memory database and create new memories that could help answer the following question: {trial["question"]}. You are NOT supposed to answer the question directly, nor will you receive any immediate reward for answering it. Instead, your goal is to create informative and relevant memories using the tools provided. Later, your reward will be determined based on how useful the memories you created are for answering the target question. Remember: during evaluation, only the memories you create now will be available as context (existing memories found with search_memory will NOT be available). Focus on creating high-quality, relevant memories that would enable someone else to answer the question using only those.**

**IMPORTANT**: The correct answer to the target question is hidden in the memory database and may require up to multiple memories across sessions to be answered correctly, and you shouldn't create fake memories that you come up with if the answer to the question is unclear from search results.

Now begin exploiting the memory database and creating new memories using the tools provided above and output the string "DONE" when you believe the task is complete."""

            # IMPORTANT: verl expects these exact keys in this format
            record = {
                # verl expects 'prompt' field with list of messages (not JSON string!)
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                "data_source": f"locomo-{trial['sample_id']}",
                "agent_name": "tool_agent",  # CRITICAL: This enables tool agent loop instead of single_turn_agent
                "extra_info": {
                    "index": idx,
                    "need_tools_kwargs": True,  # CRITICAL: This enables tool usage
                    "tools_kwargs": {
                        # 🔧 NEW: Use sample_id to reference initial memories in MemoryBrokerActor
                        # Each QA pair gets its own namespace for memory isolation
                        "search_memory": {
                            "create_kwargs": {
                                "sample_id": trial["sample_id"],  # e.g., "conv-48"
                                "namespace": trial["trajectory_id"],  # e.g., "conv-48-qa2-abc123"
                            },
                            "execute_kwargs": {
                                "sample_id": trial["sample_id"],
                                "namespace": trial["trajectory_id"],
                            },
                        },
                        "manage_memory": {
                            "create_kwargs": {
                                "sample_id": trial["sample_id"],
                                "namespace": trial["trajectory_id"],
                            },
                            "execute_kwargs": {
                                "sample_id": trial["sample_id"],
                                "namespace": trial["trajectory_id"],
                            },
                        },
                    },
                    # Data for reward computation
                    "target_question": trial["question"],
                    "target_answer": trial["answer"],
                    "namespace": trial["trajectory_id"],
                    "evidence": trial.get("evidence", []),
                    "category": trial.get("category", 0),
                    "sample_id": trial["sample_id"],
                    # Data level flags
                    "include_observation": self.include_observation,
                    "include_conversation": self.include_conversation,
                    "include_event_summary": self.include_event_summary,
                },
            }
            training_data.append(record)

        return training_data

    def _create_training_prompt(
        self, memories: List[Dict], target_question: str, conversation_context: Dict
    ) -> str:
        """Create training prompt for memory update agent."""

        # Format initial memories
        memory_text = "Initial Memory Database:\\n"
        for i, mem in enumerate(memories, 1):
            memory_text += f"{i}. [{mem['memory_type']}] {mem['content']} (from {mem['speaker']})\\n"

        # Create context about conversation
        context_text = f"\\nConversation Context:\\n"
        context_text += f"Number of sessions: {len([k for k in conversation_context.keys() if k.startswith('session')]) // 2}\\n"

        prompt = f"""You are a memory management agent tasked with optimizing a memory database to better answer questions.

{memory_text}

{context_text}

Target Question: {target_question}

Your goal is to use the available memory tools (search, manage, sample) to improve the memory database so that it can better answer the target question. You have a maximum of 30 tool calls to optimize the memory database.

Consider:
1. Are there redundant memories that should be merged?
2. Are there complex memories that should be split into focused parts?
3. Are there missing connections or inferences that should be added?
4. Are there low-quality memories that should be deleted?
5. Would sampling help identify patterns for better organization?

Begin optimizing the memory database now."""

        return prompt

    def save_parquet(self, data: List[Dict], output_path: str):
        """Save processed data to parquet format for verl."""
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(data)} examples to {output_path}")

    def save_parquet_files(self, output_dir: str = "/workspace/memupdate/data/locomo"):
        """Save training data as parquet files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Load and split data
        train_conversations, test_conversations = self.create_train_test_split()

        # Extract QA pairs
        train_trials = self.extract_qa_pairs(train_conversations)
        test_trials = self.extract_qa_pairs(test_conversations)

        # Convert to verl format
        train_data = self.create_verl_training_data(train_trials)
        test_data = self.create_verl_training_data(test_trials)

        # Save as parquet - need to serialize complex nested structures
        # Even though verl expects native objects, parquet can't handle deeply nested dicts
        # So we serialize to JSON and will need to handle deserialization in verl
        import json

        def serialize_for_parquet(data):
            """Serialize only the problematic nested fields."""
            result = []
            for record in data:
                record_copy = record.copy()
                # Only serialize extra_info which has deep nesting
                if "extra_info" in record_copy:
                    record_copy["extra_info"] = json.dumps(record_copy["extra_info"])
                # Keep prompt and messages as native lists - verl can handle these
                result.append(record_copy)
            return result

        train_data_serialized = serialize_for_parquet(train_data)
        test_data_serialized = serialize_for_parquet(test_data)

        # Use pandas for saving
        train_df = pd.DataFrame(train_data_serialized)
        test_df = pd.DataFrame(test_data_serialized)

        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

        print(
            f"✅ Saved {len(train_data)} training samples to {output_dir}/train.parquet"
        )
        print(f"✅ Saved {len(test_data)} test samples to {output_dir}/test.parquet")


def main():
    """Run preprocessing from command line."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/workspace/memupdate/data/locomo", help="Output directory")
    parser.add_argument("--input", default="/workspace/memupdate/data/locomo10.json")
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

    processor = LoCoMoProcessor(
        args.input,
        include_observation=args.include_observation,
        include_conversation=args.include_conversation,
        include_event_summary=args.include_event_summary
    )
    
    print(f"\n📋 Data levels to include:")
    print(f"  - Observations: {args.include_observation}")
    print(f"  - Conversations: {args.include_conversation}")
    print(f"  - Event Summaries: {args.include_event_summary}")
    
    processor.save_parquet_files(args.output)


if __name__ == "__main__":
    main()
