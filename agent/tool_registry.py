"""
Dynamic tool registry and management
"""
import json
from typing import Dict, List, Any, Optional, Callable
from utils.logging_config import get_logger
from utils.error_handling import ToolExecutionError

logger = get_logger(__name__)

class ToolRegistry:
    """
    Registry for managing and executing tools dynamically
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_executors: Dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        definition: Dict[str, Any],
        executor: Callable
    ):
        """
        Register a new tool
        
        Args:
            name: Tool name
            definition: Tool definition in Mistral API format
            executor: Async function to execute the tool
        """
        self.tools[name] = definition
        self.tool_executors[name] = executor
        logger.info(f"Registered tool: {name}")
    
    def unregister_tool(self, name: str):
        """Unregister a tool"""
        if name in self.tools:
            del self.tools[name]
            del self.tool_executors[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions for API calls
        
        Returns:
            List of tool definitions
        """
        return list(self.tools.values())
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool definition"""
        return self.tools.get(name)
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: str
    ) -> str:
        """
        Execute a tool with given arguments
        
        Args:
            tool_name: Name of the tool to execute
            arguments: JSON string of arguments
        
        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tool_executors:
            raise ToolExecutionError(f"Tool '{tool_name}' not found in registry")
        
        try:
            # Parse arguments
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            
            logger.info(f"Executing tool: {tool_name} with args: {args}")
            
            # Execute tool
            executor = self.tool_executors[tool_name]
            result = await executor(**args) if callable(executor) else await executor
            
            logger.info(f"Tool {tool_name} executed successfully")
            return str(result)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON arguments for tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
        except TypeError as e:
            error_msg = f"Invalid arguments for tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
    
    def get_tool_count(self) -> int:
        """Get number of registered tools"""
        return len(self.tools)
    
    def clear(self):
        """Clear all registered tools"""
        self.tools.clear()
        self.tool_executors.clear()
        logger.info("Cleared all tools from registry")
