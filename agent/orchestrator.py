"""
Main agent orchestrator with never-stop-short logic
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.mistral_client import MistralClient
from agent.tool_registry import ToolRegistry
from agent.completion_detector import CompletionDetector
from agent.task_decomposer import TaskDecomposer
from tools.web_search import WebSearchTool
from tools.code_execution import CodeExecutionTool
from tools.custom_tools import get_calculator_tool
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

class AgentOrchestrator:
    """
    Autonomous agent with comprehensive execution logic:
    - Never stops short until task is COMPLETELY done
    - Dynamic tool calling with parallel execution
    - Complete feedback loops
    - Task decomposition and subtask tracking
    - Retry logic with exponential backoff
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = MistralClient(api_key)
        self.tool_registry = ToolRegistry()
        self.completion_detector = CompletionDetector()
        self.task_decomposer = TaskDecomposer()
        self.execution_history = []
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        # Web search tool
        web_search = WebSearchTool()
        self.tool_registry.register_tool(
            web_search.name,
            web_search.get_definition(),
            web_search.execute
        )
        
        # Code execution tool
        code_exec = CodeExecutionTool()
        self.tool_registry.register_tool(
            code_exec.name,
            code_exec.get_definition(),
            code_exec.execute
        )
        
        # Calculator tool
        calculator = get_calculator_tool()
        self.tool_registry.register_tool(
            calculator.name,
            calculator.get_definition(),
            calculator.execute
        )
        
        logger.info(f"Registered {self.tool_registry.get_tool_count()} default tools")
    
    async def execute_task(
        self,
        task_description: str,
        model: str = None,
        max_iterations: int = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Main execution loop - NEVER stops until task is complete
        """
        model = model or settings.DEFAULT_MODEL
        max_iterations = max_iterations or settings.MAX_ITERATIONS
        
        # Initialize execution state
        execution = {
            "task": task_description,
            "model": model,
            "start_time": datetime.now().isoformat(),
            "iterations": [],
            "subtasks": [],
            "tool_calls": [],
            "status": "in_progress",
            "completion_confidence": 0.0
        }
        
        # Decompose task into subtasks
        subtasks = await self.task_decomposer.decompose(task_description)
        execution["subtasks"] = subtasks
        
        logger.info(f"[Orchestrator] Starting task with {len(subtasks)} subtasks")
        logger.info(f"[Orchestrator] Subtasks: {[s['description'] for s in subtasks]}")
        
        # Main execution loop
        iteration = 0
        messages = [{"role": "user", "content": task_description}]
        
        while iteration < max_iterations:
            iteration += 1
            iteration_start = datetime.now()
            
            logger.info(f"[Orchestrator] === ITERATION {iteration}/{max_iterations} ===")
            
            # Get available tools
            tools = self.tool_registry.get_tool_definitions()
            
            # Make API call with tools
            try:
                response = await self.client.chat_completion(
                    messages=messages,
                    model=model,
                    tools=tools if tools else None,
                    parallel_tool_calls=settings.ENABLE_PARALLEL_TOOLS,
                    temperature=settings.TEMPERATURE
                )
                
                # Extract assistant message
                assistant_message = response["choices"][0]["message"]
                messages.append(assistant_message)
                
                # Check for tool calls
                tool_calls = assistant_message.get("tool_calls", [])
                
                if tool_calls:
                    logger.info(f"[Orchestrator] {len(tool_calls)} tool calls requested")
                    
                    # Execute tools (parallel if independent)
                    tool_results = await self._execute_tools(tool_calls)
                    
                    # Add tool results to messages
                    for result in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["output"]
                        })
                        
                        execution["tool_calls"].append(result)
                    
                    # Update subtask completion
                    self._update_subtask_status(execution, tool_results)
                
                # Store iteration data
                execution["iterations"].append({
                    "number": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "duration": (datetime.now() - iteration_start).total_seconds(),
                    "model_response": assistant_message.get("content"),
                    "tool_calls": len(tool_calls),
                    "reasoning": self._extract_reasoning(assistant_message)
                })
                
                # Check if task is COMPLETELY finished
                is_complete, confidence, reasoning = await self.completion_detector.check_completion(
                    task_description=task_description,
                    conversation_history=messages,
                    subtasks=execution["subtasks"],
                    tool_results=execution["tool_calls"]
                )
                
                execution["completion_confidence"] = confidence
                
                logger.info(f"[Orchestrator] Completion check: {is_complete}, confidence: {confidence:.2f}")
                logger.info(f"[Orchestrator] Reasoning: {reasoning}")
                
                # CRITICAL: Only stop if TRULY complete with high confidence
                if is_complete and confidence > 0.85:
                    logger.info("[Orchestrator] Task complete with high confidence!")
                    execution["status"] = "completed"
                    break
                
                # If not complete, ask model to continue
                if not tool_calls and assistant_message.get("content"):
                    # Model thinks it's done but we disagree - push it to continue
                    incomplete_subtasks = [s for s in execution["subtasks"] if s["status"] != "completed"]
                    if incomplete_subtasks:
                        continuation_prompt = (
                            f"The following subtasks are still incomplete:\n" +
                            "\n".join([f"- {s['description']}" for s in incomplete_subtasks]) +
                            "\n\nPlease continue working on these tasks. Use tools as needed."
                        )
                        messages.append({"role": "user", "content": continuation_prompt})
                
            except Exception as e:
                logger.error(f"[Orchestrator] Iteration {iteration} failed: {e}")
                execution["iterations"].append({
                    "number": iteration,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Retry with exponential backoff
                await asyncio.sleep(min(2 ** iteration, 30))
        
        # Finalize execution
        execution["end_time"] = datetime.now().isoformat()
        execution["total_iterations"] = iteration
        
        if execution["status"] != "completed":
            execution["status"] = "max_iterations_reached"
            logger.warning(f"[Orchestrator] Task incomplete after {iteration} iterations")
        
        self.execution_history.append(execution)
        return execution
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls with parallel execution support
        """
        results = []
        
        # Check if tools can be executed in parallel
        if settings.ENABLE_PARALLEL_TOOLS and self._can_parallelize(tool_calls):
            logger.info(f"[Orchestrator] Executing {len(tool_calls)} tools in PARALLEL")
            
            # Execute all tools concurrently
            tasks = [
                self.tool_registry.execute_tool(
                    tc["function"]["name"],
                    tc["function"]["arguments"]
                )
                for tc in tool_calls
            ]
            
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (tool_call, result) in enumerate(zip(tool_calls, parallel_results)):
                if isinstance(result, Exception):
                    logger.error(f"[Orchestrator] Tool {tool_call['function']['name']} failed: {result}")
                    result_content = f"Error: {str(result)}"
                    status = "error"
                else:
                    result_content = result
                    status = "success"
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_call["function"]["name"],
                    "input": tool_call["function"]["arguments"],
                    "output": result_content,
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                })
        else:
            # Execute tools sequentially
            logger.info(f"[Orchestrator] Executing {len(tool_calls)} tools SEQUENTIALLY")
            
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                
                # Execute with retry logic
                result = await self._execute_tool_with_retry(tool_name, arguments)
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "input": arguments,
                    "output": result["output"],
                    "status": result["status"],
                    "retry_count": result.get("retry_count", 0),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    async def _execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute tool with exponential backoff retry
        """
        for attempt in range(max_retries):
            try:
                result = await self.tool_registry.execute_tool(tool_name, arguments)
                return {
                    "output": result,
                    "status": "success",
                    "retry_count": attempt
                }
            except Exception as e:
                logger.warning(f"[Orchestrator] Tool {tool_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "output": f"Tool execution failed after {max_retries} attempts: {str(e)}",
                        "status": "error",
                        "retry_count": attempt + 1
                    }
    
    def _can_parallelize(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """
        Determine if tool calls are independent and can run in parallel
        """
        # Simple heuristic: check if tools don't depend on each other
        tool_names = [tc["function"]["name"] for tc in tool_calls]
        
        # Don't parallelize if same tool called multiple times
        if len(tool_names) != len(set(tool_names)):
            return False
        
        return True
    
    def _update_subtask_status(
        self,
        execution: Dict[str, Any],
        tool_results: List[Dict[str, Any]]
    ):
        """
        Update subtask completion status based on tool results
        """
        for subtask in execution["subtasks"]:
            if subtask["status"] == "completed":
                continue
            
            # Check if any tool result addresses this subtask
            for result in tool_results:
                if self._result_addresses_subtask(result, subtask):
                    subtask["status"] = "completed"
                    subtask["completed_at"] = datetime.now().isoformat()
                    logger.info(f"[Orchestrator] Subtask completed: {subtask['description']}")
                    break
    
    def _result_addresses_subtask(
        self,
        result: Dict[str, Any],
        subtask: Dict[str, Any]
    ) -> bool:
        """
        Heuristic to check if tool result addresses a subtask
        """
        # Simple keyword matching
        subtask_keywords = set(subtask["description"].lower().split())
        result_text = str(result.get("output", "")).lower()
        
        matching_keywords = sum(1 for kw in subtask_keywords if kw in result_text)
        return matching_keywords >= len(subtask_keywords) * 0.5
    
    def _extract_reasoning(self, message: Dict[str, Any]) -> str:
        """
        Extract reasoning/thinking from model response
        """
        content = message.get("content", "")
        
        if not content:
            return "No explicit reasoning provided"
        
        # Look for reasoning patterns
        if "because" in content.lower():
            return content
        if "reasoning:" in content.lower():
            return content.split("reasoning:")[-1].strip()
        
        return content[:200] if content else "No explicit reasoning provided"
    
    async def close(self):
        """Clean up resources"""
        await self.client.close()
