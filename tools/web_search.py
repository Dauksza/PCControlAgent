"""
Mistral's built-in web_search connector
"""
from typing import Dict, Any
from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)

class WebSearchTool(BaseTool):
    """
    Use Mistral's native web_search connector or standalone implementation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "web_search"
        self.description = "Search the web for current information"
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Return tool definition for Mistral API
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information using a search engine",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (1-10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, query: str, num_results: int = 5) -> str:
        """
        Execute web search
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            Formatted search results
        """
        try:
            from duckduckgo_search import DDGS
            
            logger.info(f"Searching web for: {query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted = [f"Search results for '{query}':\n"]
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )
            
            return "\n".join(formatted)
            
        except ImportError:
            logger.error("duckduckgo-search not installed")
            return "Error: Web search functionality requires 'duckduckgo-search' package. Install with: pip install duckduckgo-search"
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {str(e)}"
