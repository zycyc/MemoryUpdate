"""Memory search tool - wraps LangMem search functionality in verl BaseTool interface."""

import logging
import os
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)

# Set up debug logging to file if requested
def _debug_log(message: str):
    """Log debug message to file if MEMUPDATE_TOOL_DEBUG is set."""
    if os.getenv('MEMUPDATE_TOOL_DEBUG'):
        log_file = os.getenv('MEMUPDATE_LOG_FILE', '/workspace/memupdate/tool_debug.log')
        try:
            with open(log_file, 'a') as f:
                f.write(f"[SearchMemoryTool] {message}\n")
                f.flush()
        except:
            pass
    print(f"[TOOL] {message}")


class SearchMemoryTool(BaseTool):
    """Memory search tool that wraps LangMem search functionality."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        # _debug_log("🔧 Initializing SearchMemoryTool...")
        
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager
        
        # Store sample_id per instance for execution-time initialization
        self._instance_sample_ids = {}  # instance_id -> sample_id
        
        # Set availability flag (referenced elsewhere in codebase)
        self.langmem_search = "available"
        
        # _debug_log(f"✅ SearchMemoryTool initialized using MemoryStoreManager")


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory search."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search_memory",
                description="Search across 3 levels of memory: conversation (raw dialogue), observation (extracted facts), and event_summary (high-level events). Multi-value filters (e.g., 'conversation,observation') search all combinations, merge results, and return top-k by semantic relevance. Complex filters create Cartesian products - source='X,Y' + speaker='A,B' searches (X,A), (X,B), (Y,A), (Y,B) and returns the globally most relevant top_k memories.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Natural language query to search for relevant memories. When list_all is True, query is ignored and memories are returned in temporal order."
                        ),
                        "top_k": OpenAIFunctionPropertySchema(
                            type="integer", 
                            description="Number of most relevant memories to retrieve. When list_all is True, top_k is ignored and all matching memories are returned."
                        ),
                        "source_filter": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Filter by source level. Examples: 'conversation' (single), 'conversation,observation' (multiple searches merged and ranked). Leave empty to search all sources."
                        ),
                        "speaker_filter": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Filter by speaker(s). Examples: 'Alice' (single), 'Alice,Bob' (searches both, returns top-k across all). Leave empty to search all speakers."
                        ),
                        "session_filter": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Filter by session(s). Examples: 'session_1' (single), 'session_1,session_2' (searches both, returns top-k across all). Leave empty to search all sessions."
                        ),
                        "list_all": OpenAIFunctionPropertySchema(
                            type="boolean",
                            description="When true, returns ALL memories matching the filters (query and top_k are ignored). Useful for exploring entire sessions or specific contexts. Use with caution because this will flood the context window. Default: false"
                        ),
                    },
                    required=["query", "top_k"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a search tool instance."""
        from uuid import uuid4
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Extract namespace and sample_id from create_kwargs
        create_kwargs = kwargs.get('create_kwargs', {})
        trial_namespace = create_kwargs.get('trial_namespace', instance_id)
        sample_id = create_kwargs.get('sample_id')
        
        # Store sample_id for this instance (needed during execute)
        if sample_id:
            self._instance_sample_ids[instance_id] = sample_id
        
        # Register the mapping from instance_id to intended namespace if different
        if trial_namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, trial_namespace)

        # Don't initialize store here - will be done during execute() if needed
        return instance_id, ToolResponse(text=f"Memory search tool ready for namespace '{trial_namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory search operation."""
        try:
            # Validate parameters type to prevent unhashable type errors
            if not isinstance(parameters, dict):
                return ToolResponse(text=f"Error: Invalid parameters type: {type(parameters)}"), 0.0, {}
            
            # Get trial_namespace directly from kwargs (passed from execute_kwargs in tool_agent_loop)
            trial_namespace = kwargs.get("trial_namespace", instance_id)
            
            query = parameters.get("query", "")
            top_k = parameters.get("top_k", None)  # None when not specified
            source_filter = parameters.get("source_filter", "")
            speaker_filter = parameters.get("speaker_filter", "")
            session_filter = parameters.get("session_filter", "")
            list_all = parameters.get("list_all", False)
            
            # Log search request concisely (only if filters are used)
            if source_filter or speaker_filter or session_filter:
                filters = []
                if source_filter:
                    filters.append(f"src:{source_filter}")
                if speaker_filter:
                    filters.append(f"spk:{speaker_filter}")
                if session_filter:
                    filters.append(f"ses:{session_filter}")
                # print(f"🔍 Search: {' | '.join(filters)}")

            # Determine effective limit based on list_all
            if list_all:
                # List all mode: use unlimited results
                effective_limit = None  # Will be handled in backend
                query = ""
            else:
                # Normal mode: use top_k or default
                effective_limit = top_k if top_k is not None else 5
                if not query:
                    return ToolResponse(text="Error: Query is required for memory search (unless using list_all=True)"), 0.0, {}

            # Use MemoryStoreManager Ray Actor for memory operations
            try:
                # Uncomment for debugging memory counts by source:
                # current_memories = await self.store_manager.get_current_memory_async(trial_namespace)
                # if current_memories:
                #     source_counts = {}
                #     for mem in current_memories:
                #         source = mem.get("metadata", {}).get("source", "unknown")
                #         source_counts[source] = source_counts.get(source, 0) + 1
                #     print(f"📊 Memory sources: {source_counts}")

                # Use Ray Actor method directly with multi-field filtering
                result = await self.store_manager.search_memory_via_actor_async(
                    trial_namespace, query, effective_limit, source_filter, speaker_filter, session_filter, list_all
                )

                if result["success"]:
                    search_results = result["results"]
                    semantic_search = result.get("semantic_search", False)
                    
                    search_type = "semantic" if semantic_search else "keyword"
                    
                    if search_results:
                        # Format results with clear memory IDs for updating
                        filter_parts = []
                        if source_filter:
                            filter_parts.append(f"source: {source_filter}")
                        if speaker_filter:
                            filter_parts.append(f"speaker: {speaker_filter}")
                        if session_filter:
                            filter_parts.append(f"session: {session_filter}")
                        filter_info = f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""
                        search_header = f"Found {len(search_results)} relevant memories using {search_type} search{filter_info}:\n"
                        formatted_results = search_header
                        returned_ids = []
                        for i, mem in enumerate(search_results, 1):
                            memory_id = mem.get("id", f"unknown_{i}")
                            returned_ids.append(memory_id)
                            content = mem.get("content", "")
                            metadata = mem.get("metadata", {})
                            
                            # Format metadata for display including timestamp
                            metadata_str = ""
                            if metadata.get("speaker"):
                                metadata_str += f" | Speaker: {metadata['speaker']}"
                            if metadata.get("evidence"):
                                metadata_str += f" | Evidence: {metadata['evidence']}"
                            if metadata.get("session"):
                                metadata_str += f" | Session: {metadata['session']}"
                            if metadata.get("timestamp"):
                                metadata_str += f" | Time: {metadata['timestamp']}"
                            if metadata.get("source"):
                                metadata_str += f" | Source: {metadata['source']}"
                            
                            formatted_results += f"{i}. ID: {memory_id} | Content: {content}{metadata_str}\n"
                        
                        # zycyc debug prints
                        # print(f"✅ [SEARCH_MEMORY] Returned {len(search_results)} memories:")
                        # print(f"📄 Found memories: {formatted_results}")
                        
                        return ToolResponse(
                            text=formatted_results
                        ), 0.0, {"memories_found": len(search_results), "search_type": search_type, "source_filter": source_filter, "speaker_filter": speaker_filter, "session_filter": session_filter}
                    else:
                        # Concise "no results" message - only show filters if used
                        if source_filter or speaker_filter or session_filter:
                            filters = []
                            if source_filter:
                                filters.append(f"src:{source_filter}")
                            if speaker_filter:
                                filters.append(f"spk:{speaker_filter}")
                            if session_filter:
                                filters.append(f"ses:{session_filter}")
                            # print(f"⚠️ No results: {' | '.join(filters)}")
                        filter_parts = []
                        if source_filter:
                            filter_parts.append(f"source: {source_filter}")
                        if speaker_filter:
                            filter_parts.append(f"speaker: {speaker_filter}")
                        if session_filter:
                            filter_parts.append(f"session: {session_filter}")
                        filter_info = f" with filters ({', '.join(filter_parts)})" if filter_parts else ""
                        return ToolResponse(
                            text=f"No memories found matching '{query}' using {search_type} search{filter_info}."
                        ), 0.0, {"memories_found": 0, "search_type": search_type, "source_filter": source_filter, "speaker_filter": speaker_filter, "session_filter": session_filter}
                else:
                    return ToolResponse(text=f"Failed to search memories: {result.get('error', 'Unknown error')}"), 0.0, {}
                
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                return ToolResponse(text=f"Memory search failed: {str(e)}"), 0.0, {"error": str(e)}

        except Exception as e:
            logger.error(f"Memory search execution failed: {e}")
            return ToolResponse(text=f"Memory search failed: {str(e)}"), 0.0, {"error": str(e)}

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        trial_namespace = kwargs.get("trial_namespace", instance_id)

        # Return success (no actual cleanup needed since MemoryStoreManager handles persistence)
        return f"Released SearchMemoryTool instance for namespace '{trial_namespace}'"