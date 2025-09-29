"""Memory management tool - wraps LangMem management functionality in verl BaseTool interface."""

import logging
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



class ManageMemoryTool(BaseTool):
    """Memory management tool that wraps LangMem management functionality."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        # _debug_log("🔧 Initializing ManageMemoryTool...")
        
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        
        # Use shared memory store manager
        from .base_memory_tool import MemoryStoreManager
        self.store_manager = MemoryStoreManager
        
        # Store sample_id per instance for execution-time initialization
        self._instance_sample_ids = {}  # instance_id -> sample_id
        
        # Set availability flag (referenced elsewhere in codebase)
        self.langmem_manage = "available"
        
        # _debug_log(f"✅ ManageMemoryTool initialized using MemoryStoreManager")


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for memory management."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="manage_memory",
                description="Create new memories in the database. This is your primary tool for adding new integrated memories",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "operation": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Type of operation: create only (update is currently disabled)",
                            enum=["create"]  # "update" temporarily disabled
                        ),
                        "content": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Memory content to store or update"
                        ),
                        "memory_id": OpenAIFunctionPropertySchema(
                            type="string",
                            description="ID of existing memory to update (REQUIRED for update operation). Must be an actual ID returned from search_memory - do not guess or invent IDs."
                        ),
                        "speaker": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The speaker who provided this memory (e.g., 'John', 'Mary', etc.)"
                        ),
                        "evidence": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Supporting evidence or references for this memory"
                        ),
                        "session": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The session context where this memory was created (e.g., 'session_1_observation')"
                        )
                    },
                    required=["operation", "content", "speaker", "evidence", "session"]
                )
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a manage tool instance."""
        from uuid import uuid4
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Extract namespace from create_kwargs
        create_kwargs = kwargs.get('create_kwargs', {})
        trial_namespace = create_kwargs.get('trial_namespace', instance_id)
        sample_id = create_kwargs.get('sample_id')
        
        # Store sample_id for this instance (needed during execute)
        if sample_id:
            self._instance_sample_ids[instance_id] = sample_id
        
        # Register the mapping from instance_id to intended namespace if different
        if trial_namespace != instance_id:
            self.store_manager.register_instance_namespace(instance_id, trial_namespace)

        # Tool creation is now passive - actual memory initialization happens during execute()
        return instance_id, ToolResponse(text=f"Memory management tool ready for namespace '{trial_namespace}'")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute memory management operation."""
        # _debug_log(f"🛠️  ManageMemoryTool.execute called with operation: {parameters.get('operation', 'N/A')}")
        try:
            # print(f"🛠️  MEMUPDATE DEBUG: ManageMemoryTool.execute kwargs: {kwargs}")
            
            # Validate parameters type to prevent unhashable type errors
            if not isinstance(parameters, dict):
                return ToolResponse(text=f"Error: Invalid parameters type: {type(parameters)}"), 0.0, {}
            
            # Get trial_namespace directly from kwargs (passed from execute_kwargs in tool_agent_loop)
            trial_namespace = kwargs.get("trial_namespace", instance_id)
            # print(f"🔧 ManageMemoryTool executing for instance_id '{instance_id}' with namespace '{trial_namespace}', kwargs: {kwargs}")

            operation = parameters.get("operation", "create")
            content = parameters.get("content", "")
            speaker = parameters.get("speaker", "")
            evidence = parameters.get("evidence", "")
            session = parameters.get("session", "")
            memory_id = parameters.get("memory_id", None)
            
            # Enhanced parameter validation with better error messages
            if not content:
                return ToolResponse(text="Error: 'content' parameter is required"), 0.0, {}
            
            if not operation or operation not in ["create", "update"]:
                return ToolResponse(text="Error: 'operation' must be either 'create' or 'update'"), 0.0, {}
            
            # Check for update operation without memory_id BEFORE attempting the operation
            if operation == "update" and not memory_id:
                return ToolResponse(text="Error: 'memory_id' is required for update operations. Use search_memory first to get valid memory IDs."), 0.0, {}
            
            # Log the manage memory request  
            # zycyc debug prints
            # print(f"📝 [MANAGE_MEMORY] Namespace: {trial_namespace} | Operation: {operation}")

            # 🔍 DEBUG: Log the operation being performed
            # _debug_log(f"📝 MEMUPDATE DEBUG: ManageMemoryTool.execute called with namespace='{namespace}', operation='{operation}',content='{content}'")

            # Additional metadata validation
            
            # Validate required metadata fields
            if not speaker:
                return ToolResponse(text="Error: Speaker is required for memory management"), 0.0, {}
            if not evidence:
                return ToolResponse(text="Error: Evidence is required for memory management"), 0.0, {}
            if not session:
                return ToolResponse(text="Error: Session is required for memory management"), 0.0, {}

            # 🔧 CRITICAL FIX: Use Ray Actor for all store operations to avoid serialization issues
            # Getting local store copies via ray.get() creates separate instances!
            
            if operation == "create":
                # Always generate sequential ID automatically - agents cannot specify create IDs
                result = await self.store_manager.create_memory_via_actor(trial_namespace, {
                    "action": "create",
                    "content": content,
                    "metadata": {
                        "type": "episodic", 
                        "source": "agent_output",
                        "speaker": speaker,
                        "evidence": evidence,
                        "session": session,
                        "timestamp": "off_task_memory_updating"  # Fixed timestamp for agent memories
                    }
                })
                
                if result["success"]:
                    created_id = result.get("memory_id", "unknown")
                    # zycyc debug prints
                    # print(f"✅ [MANAGE_MEMORY] CREATE: {created_id}, content: {content}")
                    return ToolResponse(
                        text=f"Successfully created episodic memory with ID {created_id}: {content}..."
                    ), 0.0, {"operation": operation, "memory_type": "episodic", "memory_id": created_id}
                else:
                    return ToolResponse(text=f"Failed to create memory: {result['result']}"), 0.0, {}
                    
            elif operation == "update":
                # Temporarily disabled
                return ToolResponse(text="Error: Update operation is temporarily disabled. Only 'create' is available."), 0.0, {}
            elif False:  # Original update code (temporarily disabled)
                if not memory_id:
                    print(f"❌ [MANAGE_MEMORY] Update failed: No memory_id provided")
                    return ToolResponse(text="Error: memory_id required for update operation. Use search_memory first to get valid IDs."), 0.0, {}
                
                # Log the update attempt
                # zycyc debug prints
                # print(f"🔄 [MANAGE_MEMORY] Update attempt for ID: '{memory_id}'")
                
                # First check if the memory actually exists - REJECT if not found
                current_memories = await self.store_manager.get_current_memory_async(trial_namespace)
                existing_ids = [mem.get("id") for mem in current_memories if mem.get("id")]
                
                # Find the existing memory for old vs new comparison
                existing_memory = None
                for mem in current_memories:
                    if mem.get("id") == memory_id:
                        existing_memory = mem
                        break
                
                if memory_id not in existing_ids:
                    # Enhanced error logging with detailed diagnosis
                    print(f"❌ [MANAGE_MEMORY] INVALID ID: '{memory_id}'")
                    
                    # Diagnose the type of error
                    if not memory_id.startswith('memory_'):
                        print(f"🔍 DIAGNOSIS: Wrong format - expected 'memory_N' pattern")
                    elif memory_id.startswith('memory_') and memory_id[7:].isdigit():
                        print(f"🔍 DIAGNOSIS: Valid format but ID doesn't exist in store")
                    else:
                        print(f"🔍 DIAGNOSIS: Invalid format - should be 'memory_' + number")
                    
                    # Show available options
                    if existing_ids:
                        print(f"📊 AVAILABLE IDs: {existing_ids[:8]}")
                        if len(existing_ids) > 8:
                            print(f"    ... and {len(existing_ids) - 8} more")
                    else:
                        print(f"📊 NO MEMORIES: Store is empty - create memories first")
                    
                    print(f"💡 HINT: Use search_memory() to get valid IDs")
                    
                    return ToolResponse(
                        text=f"Error: Memory ID '{memory_id}' not found. Use search_memory to get valid memory IDs. "
                    ), 0.0, {"operation": "update_failed", "reason": "invalid_memory_id"}
                
                # Only proceed with update if memory exists
                result = await self.store_manager.update_memory_via_actor(trial_namespace, {
                    "action": "update", 
                    "id": memory_id,
                    "content": content,
                    "metadata": {
                        "type": "episodic", 
                        "source": "agent_output",
                        "speaker": speaker,
                        "evidence": evidence,
                        "session": session,
                        "timestamp": "off_task_memory_updating"  # Fixed timestamp for agent memories
                    }
                })
                
                if result["success"]:
                    # Log the update with old vs new content comparison
                    # zycyc debug prints
                    # print(f"✅ [MANAGE_MEMORY] UPDATE: {memory_id}")
                    if existing_memory:
                        pass  # Memory exists, proceeding with update
                        # old_content = existing_memory.get("content", "")
                        # print(f"📝 OLD: {old_content}")
                    # print(f"📝 NEW: {content}")
                    return ToolResponse(
                        text=f"Successfully updated existing memory {memory_id}: {content}..."
                    ), 0.0, {"operation": operation, "memory_id": memory_id, "was_actual_update": True}
                else:
                    return ToolResponse(text=f"Failed to update memory: {result['result']}"), 0.0, {}
            else:
                # 🚫 Unknown operation - should only be create or update
                print(f"❌ ERROR: Unknown operation '{operation}' - valid operations are: create or update")
                return ToolResponse(text=f"Error: Unknown operation '{operation}'. Valid operations: create or update"), 0.0, {}

        except Exception as e:
            logger.error(f"Memory management execution failed: {e}")
            return ToolResponse(text=f"Memory management failed: {str(e)}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance but preserve memory state for reward computation."""
        # 🔧 CRITICAL: Don't clear namespace here - memory state needed for reward computation
        # MemoryStoreManager persists across tool instances using class-level storage
        trial_namespace = kwargs.get("trial_namespace", instance_id)
        
        # Return success (no actual cleanup needed since MemoryStoreManager handles persistence)
        return f"Released ManageMemoryTool instance for namespace '{trial_namespace}'"

