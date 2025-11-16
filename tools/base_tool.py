"""
Base tool class for all tools
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """
    Abstract base class for all tools
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "Base tool"
    
    @abstractmethod
    def get_definition(self) -> Dict[str, Any]:
        """
        Get tool definition in Mistral API format
        
        Returns:
            Tool definition dictionary
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        Execute the tool
        
        Args:
            **kwargs: Tool-specific arguments
        
        Returns:
            Tool execution result as string
        """
        pass
    
    def validate_arguments(self, **kwargs) -> bool:
        """
        Validate tool arguments
        
        Args:
            **kwargs: Arguments to validate
        
        Returns:
            True if valid, False otherwise
        """
        return True
