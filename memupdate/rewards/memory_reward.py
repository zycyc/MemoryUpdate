"""LLM-based memory reward manager for MemUpdate RL training using LLM-as-judge approach."""

import logging
from collections import defaultdict
from typing import Any, Dict, List
import asyncio
import aiohttp
import ray.train
import torch

try:
    from verl.workers.reward_manager.abstract import AbstractRewardManager
    from verl.protocol import DataProto
    from verl.workers.reward_manager import register
    VERL_AVAILABLE = True
except ImportError:
    # Fallback for when verl is not available
    class AbstractRewardManager:
        pass
    class DataProto:
        pass
    register = lambda name: lambda cls: cls
    VERL_AVAILABLE = False

from memupdate.tools.base_memory_tool import MemoryStoreManager

logger = logging.getLogger(__name__)


@register("memory_rag")
class MemoryRewardManager(AbstractRewardManager):
    """LLM-as-judge based reward manager that computes binary correct/incorrect scores."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        # Store verl parameters
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # MemUpdate-specific config from kwargs
        self.max_total_memories = kwargs.get("max_total_memories", 100)

        # Load evaluator URL from config or environment
        self.evaluator_url = self._load_evaluator_url(kwargs)

        # Load API key from local file (should be in .gitignore)
        self.api_key = self._load_api_key()
        
        print(f"✅ Initialized LLM-as-Judge MemoryRewardManager with evaluator_url={self.evaluator_url}")

    def _load_evaluator_url(self, kwargs: dict) -> str:
        """Load evaluator URL from kwargs, config file, or environment variable."""
        import os
        from pathlib import Path

        # 1. Check kwargs first (highest priority)
        if "evaluator_url" in kwargs:
            url = kwargs["evaluator_url"]
            print(f"✅ Using evaluator URL from config: {url}")
            return url

        # 2. Try loading from config file
        possible_config_paths = [
            "/workspace/memupdate/config/evaluator_config.yaml",  # In container
            os.path.join(os.path.dirname(__file__), "../../config/evaluator_config.yaml"),  # Relative to this file
            "config/evaluator_config.yaml",  # Current directory
        ]

        for config_path in possible_config_paths:
            expanded_path = os.path.expanduser(config_path)
            if os.path.exists(expanded_path):
                try:
                    import yaml
                    with open(expanded_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if config and "evaluator_url" in config:
                        url = config["evaluator_url"]
                        print(f"✅ Using evaluator URL from {expanded_path}: {url}")
                        return url
                except Exception as e:
                    print(f"⚠️ Failed to read evaluator config from {expanded_path}: {e}")
                    continue

        # 3. Try environment variable
        url = os.getenv("EVALUATOR_URL")
        if url:
            print(f"✅ Using evaluator URL from EVALUATOR_URL environment variable: {url}")
            return url

        # 4. Final fallback - fail with helpful error message
        raise ValueError(
            "No evaluator URL configured. Please set one of:\n"
            "  1. Pass evaluator_url in reward_model config\n"
            "  2. Create config/evaluator_config.yaml with evaluator_url key\n"
            "  3. Set EVALUATOR_URL environment variable\n"
            "See config/evaluator_config.yaml.example for reference."
        )

    def _load_api_key(self) -> str:
        """Load API key from environment variable or file (deprecated)."""
        import os

        # 1. Priority: Check environment variable first (recommended)
        api_key = os.getenv("LITELLM_API_KEY")
        if api_key:
            print("✅ Loaded API key from LITELLM_API_KEY environment variable")
            return api_key

        # 2. Fallback: Try local file for backwards compatibility (deprecated)
        possible_paths = [
            "/workspace/memupdate/api_key.txt",  # In container
            os.path.join(os.path.dirname(__file__), "../../api_key.txt"),  # Relative to this file
            "api_key.txt",  # Current directory
        ]

        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                try:
                    with open(expanded_path, 'r') as f:
                        api_key = f.read().strip()
                    if api_key:
                        print(f"⚠️  Loaded API key from {expanded_path} (deprecated - use LITELLM_API_KEY env var instead)")
                        return api_key
                except Exception as e:
                    print(f"⚠️ Failed to read API key from {expanded_path}: {e}")
                    continue

        # Final fallback - error with clear instructions
        raise ValueError(
            "No API key found. Please set LITELLM_API_KEY in your .env file.\n"
            "See .env.example for configuration template."
        )

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Main entry point for reward computation using LLM-as-judge."""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
            
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Process all batch items concurrently for efficiency
        tasks = []
        for i in range(len(data)):
            tasks.append(self._process_single_item(data[i], i))
        
        # Run all reward computations in parallel
        results = asyncio.run(self._run_concurrent_rewards(tasks))
        
        # Process results
        for i, (episode_reward, extra_info) in enumerate(results):
            data_item = data[i]
            
            # Decode response for validation length
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            # Set reward at the end of the valid response
            reward_tensor[i, valid_response_length - 1] = episode_reward
            
            # Store extra info
            for key, value in extra_info.items():
                reward_extra_info[key].append(value)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    async def _run_concurrent_rewards(self, tasks):
        """Run reward computation tasks concurrently."""
        return await asyncio.gather(*tasks)

    async def _process_single_item(self, data_item, item_index):
        """Process a single batch item for reward computation."""
        try:
            # Extract memory states from extra_info
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            # Get identifiers for memory retrieval
            sample_id = extra_info.get("sample_id")  # e.g., "conv-48"
            if not sample_id:
                print(f"⚠️ No sample_id found in extra_info")
                
            target_question = extra_info.get("target_question", "")
            target_answer = extra_info.get("target_answer", "")
            
            # Get trial namespace for accessing modified memories
            trial_namespace = extra_info.get("trial_namespace")
            
            # Compute reward for this episode
            episode_reward, performance_old, performance_new = await self.compute_single_reward_async(
                target_question,
                target_answer,
                trial_namespace,  # Pass namespace for retrieval
                sample_id  # Pass sample_id for raw memory evaluation
            )
            
            # Clean up memory store after reward computation (fire-and-forget)
            if trial_namespace:
                # Fire-and-forget cleanup - don't await it to avoid blocking reward computation
                asyncio.create_task(MemoryStoreManager.cleanup_conversation_async(trial_namespace))

            # Prepare extra info
            extra_info_result = {
                "memory_reward": episode_reward,
                "performance_old": performance_old,
                "performance_new": performance_new,
                "target_question": target_question,
                "target_answer": target_answer,
                "category": extra_info.get("category", 0),  # Pass through category information
            }
            
            return episode_reward, extra_info_result
            
        except Exception as e:
            logger.error(f"Error computing reward for batch item {item_index}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return 0 reward and error info
            return 0.0, {
                "memory_reward": 0.0,
                "performance_old": 0.0,
                "performance_new": 0.0,
                "error": str(e),
                "target_question": "",
                "target_answer": "",
                "category": 0,  # Default category for errors
            }

    async def compute_single_reward_async(
        self,
        target_question: str,
        target_answer: str,
        trial_namespace: str = None,
        sample_id: str = None,
    ) -> tuple[float, float, float]:
        """
        Compute reward using LLM-as-judge approach.
        
        reward = correct_new (1 or 0) - correct_old (1 or 0)
        
        Args:
            target_question: Question to evaluate
            target_answer: Ground truth answer
            trial_namespace: Namespace for trial-specific memories (agent-modified)
            sample_id: Raw sample_id for accessing unmodified initial memories
        """
        try:
            # Compute QA performance using LLM-as-judge
            # For initial memories, use sample_id to access raw unmodified store
            # For final memories, use trial_namespace to access agent-modified store
            tasks = [
                self.evaluate_single_qa_llm(target_question, target_answer, sample_id, "initial"),  # Use sample_id for raw memories
                self.evaluate_single_qa_llm(target_question, target_answer, trial_namespace, "final")  # Use trial_namespace for modified memories
            ]

            result_old, result_new = await asyncio.gather(*tasks)
            
            debug_info = None
            # if old got right but new got wrong, print debug info (only on rank 0 to avoid interleaved logs)
            if ray.train.get_context().get_world_rank() == 0:
                # if old got right but new got wrong, print debug info
                if result_old["is_correct"] and not result_new["is_correct"]:
                    debug_info = f"\n\nTarget question: {target_question}\n✅ [LLM-as-Judge Evaluation Debug - initial memories]\n✅ Context:\n{result_old['context']}\n✅ Generated Answer: {result_old['generated_answer']}\n✅ Golden Answer: {result_old['golden_answer']}\n✅ Is Generated Answer Correct (LLM judge): {result_old['is_correct']}\n❌ [LLM-as-Judge Evaluation Debug - final memories]\n❌ Context:\n{result_new['context']}\n❌ Generated Answer: {result_new['generated_answer']}\n❌ Golden Answer: {result_new['golden_answer']}\n❌ Is Generated Answer Correct (LLM judge): {result_new['is_correct']}\n\n"
                # if old got wrong but new got right, print debug info
                elif not result_old["is_correct"] and result_new["is_correct"]:
                    debug_info = f"\n\nTarget question: {target_question}\n❌ [LLM-as-Judge Evaluation Debug - initial memories]\n❌ Context:\n{result_old['context']}\n❌ Generated Answer: {result_old['generated_answer']}\n❌ Golden Answer: {result_old['golden_answer']}\n❌ Is Generated Answer Correct (LLM judge): {result_old['is_correct']}\n✅ [LLM-as-Judge Evaluation Debug - final memories]\n✅ Context:\n{result_new['context']}\n✅ Generated Answer: {result_new['generated_answer']}\n✅ Golden Answer: {result_new['golden_answer']}\n✅ Is Generated Answer Correct (LLM judge): {result_new['is_correct']}\n\n"
                if debug_info:
                    print(debug_info)
                

            actual_reward = result_new["reward"] + result_new["format_reward"]
            performance_old = result_old["reward"]
            performance_new = result_new["reward"]

            return actual_reward, performance_old, performance_new

        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return -0.1  # Small penalty for errors

    async def evaluate_single_qa_llm(self, question: str, golden_answer: str, namespace: str = None, memory_state: str = "") -> dict:
        """
        Evaluate QA performance using LLM-as-judge approach.
        
        Returns 1.0 if the LLM judge determines the answer is correct based on retrieved memories,
        0.0 if incorrect.
        
        Args:
            question: Question to evaluate
            golden_answer: Ground truth answer
            namespace: Either sample_id (for initial) or trial_namespace (for final)
            memory_state: "initial" or "final" to determine evaluation mode
        """
        try:
            # 1. Retrieve relevant memories using the same method as tools
            if memory_state == "initial":
                # For initial memories, namespace is the raw sample_id
                # This searches in the unmodified raw memory store
                context_memories = await self._rag_retrieve_async(question, top_k=5, trial_namespace=namespace)
                format_reward = 0.0  # no bonus/penalty for initial
            elif memory_state == "final":
                # For final memories, namespace is the trial_namespace
                # This searches only agent_output memories in the modified store
                context_memories = await self._rag_retrieve_async(question, top_k=5, trial_namespace=namespace, source_filter="agent_output", list_all=True)
                if context_memories:
                    format_reward = 0.1 # small bonus for having any agent_output memories in final
                else:
                    format_reward = -1
            else:
                print(f"⚠️ Unknown memory_state '{memory_state}'")
                
            # 2. Build context string
            if not context_memories:
                context = "No relevant information found."
            else:
                context = "\n".join([
                    f"Memory {i+1}: {mem.get('content', '')}" 
                    for i, mem in enumerate(context_memories)
                ])
            
            # 3. Generate answer using the model
            generated_answer = await self._generate_answer_from_context(context, question)
            
            # 4. Create prompt for LLM-as-judge evaluation  
            prompt = self._build_llm_judge_prompt(question, generated_answer, golden_answer)
            
            # 5. Get binary correctness judgment from LLM
            is_correct = await self._get_llm_judgment(prompt)

            reward = 1.0 if is_correct else 0.0
            
            result = {"reward": reward, 
                      "format_reward": format_reward,
                      "question": question,
                      "context": context,
                      "generated_answer": generated_answer,
                      "golden_answer": golden_answer,
                      "is_correct": is_correct,
                      }
            return result

        except Exception as e:
            logger.error(f"LLM-as-judge evaluation failed: {e}")
            return 0.0, 0.0  # Failure defaults to incorrect

    async def _generate_answer_from_context(self, context: str, question: str) -> str:
        """Generate answer from context using the model."""
        answer_prompt = f"""You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to any timestamps or temporal information
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent information
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on context. For example, if a memory mentions 
   "went to India last year," determine when that occurred.
6. Always convert relative time references to specific dates, months, or years when possible
7. Focus only on the content of the memories provided
8. The answer should be concise and direct, less than 5-6 words when possible

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Look for explicit mentions of dates, times, locations, or events that answer the question
3. If the answer requires calculation, show your work
4. Formulate a precise, concise answer based solely on the evidence in the memories
5. Double-check that your answer directly addresses the question asked
6. Ensure your final answer is specific and avoids vague time references

Memories:

{context}

Question: {question}

Answer:"""

        # Retry logic for slow server
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=120)  # 120s timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    payload = {
                        "model": "openai/gpt-oss-120b",
                        "messages": [{"role": "user", "content": answer_prompt}],
                        "max_tokens": 1024,  # Reduced from 9999 to 1024 for faster generation
                        "temperature": 0.0,
                    }
                    
                    async with session.post(
                        f"{self.evaluator_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Answer generation API error {response.status}: {error_text}")
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return "Error generating answer"
                        
                        result = await response.json()
                        
                        if "choices" not in result or not result["choices"]:
                            logger.error("No choices in answer generation response")
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return "No answer generated"
                        
                        choice = result["choices"][0]
                        generated_text = choice.get("message", {}).get("content", "").strip()
                        
                        # Extract final channel content if it's in Harmony format
                        import re
                        final_pattern = r'<\|channel\|>final.*?<\|message\|>(.*?)(?:<\||$)'
                        match = re.search(final_pattern, generated_text, re.DOTALL)
                        
                        if match:
                            return match.group(1).strip()
                        else:
                            return generated_text
                            
            except asyncio.TimeoutError:
                logger.error(f"Answer generation timeout (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return "Timeout generating answer"
            except Exception as e:
                logger.error(f"Error generating answer (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return "Error generating answer"
        
        return "Failed after all retries"

    def _build_llm_judge_prompt(self, question: str, generated_answer: str, golden_answer: str) -> str:
        """Build prompt for LLM-as-judge evaluation using exact memobase implementation."""
        prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question
(2) a 'gold' (ground truth) answer
(3) a generated answer

Please be generous in your evaluation - if the answer is on the right topic and aligned with the expected information, consider it correct even if it's not perfectly worded.

Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short explanation, then finish with CORRECT or WRONG.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""
        return prompt

    async def _get_llm_judgment(self, prompt: str) -> bool:
        """Get binary correctness judgment from LLM-as-judge."""
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=120)  # 120s timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    payload = {
                        "model": "openai/gpt-oss-120b",  # Use same model as test_correct_vs_incorrect.py
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024,  # Reduced from 9999 to 1024 for judgment task
                        "temperature": 0.0,  # Deterministic judgment
                    }
                    
                    async with session.post(
                        f"{self.evaluator_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"LLM API error {response.status} (attempt {attempt + 1}/{max_retries + 1}): {error_text}")
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return False
                        
                        result = await response.json()
                        
                        if "choices" not in result or not result["choices"]:
                            logger.error(f"No choices in LLM response (attempt {attempt + 1}/{max_retries + 1})")
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return False
                        
                        choice = result["choices"][0]
                        generated_text = choice.get("message", {}).get("content", "").strip()
                        
                        # Parse the JSON response to extract judgment
                        is_correct = self._parse_judgment_response(generated_text)
                        
                        return is_correct
                        
            except asyncio.TimeoutError:
                logger.error(f"LLM API timeout (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
            except Exception as e:
                logger.error(f"Error getting LLM judgment (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
        
        return False

    def _parse_judgment_response(self, response: str) -> bool:
        """Parse the LLM judge response, handling Harmony format."""
        import json
        import re
        
        try:
            # First, try to extract JSON from the final channel
            # Handle both formats: <|channel|>final<|message|> and <|channel|>final <|constrain|>json<|message|>
            final_pattern = r'<\|channel\|>final.*?<\|message\|>(.*?)(?:<\||$)'
            match = re.search(final_pattern, response, re.DOTALL)
            
            if match:
                final_content = match.group(1).strip()
                # Try to parse JSON from final channel content
                try:
                    json_data = json.loads(final_content)
                    label = json_data["label"]
                    # Handle case where label might be a dict or other type
                    if isinstance(label, dict):
                        # If label is a dict, look for actual label value
                        label = label.get("label", str(label))
                    label_str = str(label).upper() if label is not None else ""
                    return 1 if label_str == "CORRECT" else 0
                except json.JSONDecodeError:
                    # Look for JSON within the final content
                    json_match = re.search(r'\{.*?\}', final_content, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group(0))
                        label = json_data["label"]
                        # Handle case where label might be a dict or other type
                        if isinstance(label, dict):
                            label = label.get("label", str(label))
                        label_str = str(label).upper() if label is not None else ""
                        return 1 if label_str == "CORRECT" else 0
            
            # Fallback: try to parse entire response as JSON
            json_data = json.loads(response)
            label = json_data["label"]
            # Handle case where label might be a dict or other type
            if isinstance(label, dict):
                label = label.get("label", str(label))
            label_str = str(label).upper() if label is not None else ""
            return 1 if label_str == "CORRECT" else 0
            
        except Exception as e:
            logger.error(f"Error parsing judgment response: {e}")
            # Fallback: look for keywords in the response
            # Handle case where response might be a dict or other type
            response_str = str(response) if not isinstance(response, str) else response
            response_upper = response_str.upper()
            if "CORRECT" in response_upper and "WRONG" not in response_upper:
                return True
            elif "WRONG" in response_upper and "CORRECT" not in response_upper:
                return False
            
            logger.warning(f"Ambiguous judgment response: {response}")
            return False

    async def _rag_retrieve_async(self, question: str, top_k: int = 5, trial_namespace: str = None, source_filter: str = "", list_all: bool = False) -> List[Dict]:
        """
        Retrieve top-k relevant memories using the same search as tools.
        """
        # Use semantic search if we have a namespace (same as reward manager)
        if trial_namespace:
            try:
                from memupdate.tools.base_memory_tool import MemoryStoreManager

                # Use Ray Actor search (same as current reward manager)
                result = await MemoryStoreManager.search_memory_via_actor_async(
                    trial_namespace=trial_namespace,
                    query=question,
                    limit=top_k,
                    source_filter=source_filter,
                    list_all=list_all
                )
                
                if result["success"] and result["results"]:
                    return result["results"]
                else:
                    print(f"⚠️ Semantic search failed in LLM reward manager, returning empty context")
                    return []

            except Exception as e:
                print(f"⚠️ Semantic search error in LLM reward manager ({e}), returning empty context")
                return []

        # Fallback when no namespace provided
        print("⚠️ No namespace for semantic search, returning empty context")
        return []

    def _rag_retrieve_keyword(self, memory_db: List[Dict], question: str, top_k: int = 5) -> List[Dict]:
        """
        Keyword-based memory retrieval (fallback method).
        """
        if not memory_db:
            return []
            
        question_words = set(question.lower().split())
        
        scored_memories = []
        for memory in memory_db:
            content = memory.get("content", "").lower()
            memory_words = set(content.split())
            
            # Compute simple overlap score
            overlap = len(question_words.intersection(memory_words))
            if overlap > 0:
                score = overlap / max(len(question_words), len(memory_words))
                scored_memories.append((score, memory))
        
        # Sort by relevance and return top-k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored_memories[:top_k]]
