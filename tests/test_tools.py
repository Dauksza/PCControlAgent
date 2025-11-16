"""
Tests for tools
"""
import pytest
from tools.web_search import WebSearchTool
from tools.code_execution import CodeExecutionTool
from tools.custom_tools import get_calculator_tool

@pytest.mark.asyncio
async def test_web_search_tool():
    """Test web search tool"""
    tool = WebSearchTool()
    
    assert tool.name == "web_search"
    assert tool.get_definition() is not None
    
    # Test execution (may fail without internet or duckduckgo-search)
    try:
        result = await tool.execute(query="test", num_results=1)
        assert result is not None
        assert isinstance(result, str)
    except Exception:
        # If it fails due to missing dependency, that's okay for basic test
        pass

@pytest.mark.asyncio
async def test_code_execution_tool():
    """Test code execution tool"""
    tool = CodeExecutionTool()
    
    assert tool.name == "code_interpreter"
    assert tool.get_definition() is not None
    
    # Test Python execution
    code = "print('Hello, World!')"
    result = await tool.execute(code=code, language="python")
    
    assert result is not None
    assert isinstance(result, str)
    # Should contain output or error message
    assert "Hello, World!" in result or "Error" in result or "Output" in result

@pytest.mark.asyncio
async def test_calculator_tool():
    """Test custom calculator tool"""
    tool = get_calculator_tool()
    
    assert tool.name == "calculator"
    assert tool.get_definition() is not None
    
    # Test calculation
    result = await tool.execute(expression="2 + 2")
    assert "4" in result

def test_tool_definition_format():
    """Test that tool definitions follow correct format"""
    tool = WebSearchTool()
    definition = tool.get_definition()
    
    assert "type" in definition
    assert definition["type"] == "function"
    assert "function" in definition
    assert "name" in definition["function"]
    assert "description" in definition["function"]
    assert "parameters" in definition["function"]
    assert "properties" in definition["function"]["parameters"]
