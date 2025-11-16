"""
Custom tools defined by users
"""
from typing import Dict, Any, Callable
from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)

class CustomTool(BaseTool):
    """
    Wrapper for user-defined custom tools
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        executor: Callable
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.parameters = parameters
        self.executor = executor
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Return tool definition for Mistral API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    async def execute(self, **kwargs) -> str:
        """
        Execute custom tool
        
        Args:
            **kwargs: Tool-specific arguments
        
        Returns:
            Execution result
        """
        logger.info(f"Executing custom tool: {self.name}")
        
        try:
            result = await self.executor(**kwargs) if callable(self.executor) else self.executor
            return str(result)
        except Exception as e:
            logger.error(f"Custom tool execution failed: {e}")
            return f"Error executing {self.name}: {str(e)}"


# Example custom tool functions

async def calculator_tool(expression: str) -> str:
    """
    Simple calculator tool
    """
    try:
        # Use eval cautiously - in production, use a safe expression evaluator
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_calculator_tool() -> CustomTool:
    """
    Get calculator custom tool
    """
    return CustomTool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        },
        executor=calculator_tool
    )
