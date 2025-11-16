"""
Tests for agent orchestrator
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from agent.orchestrator import AgentOrchestrator

@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing"""
    return AgentOrchestrator(api_key="test_key")

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization"""
    assert orchestrator.client is not None
    assert orchestrator.tool_registry is not None
    assert orchestrator.completion_detector is not None
    assert orchestrator.task_decomposer is not None
    
    # Check default tools are registered
    assert orchestrator.tool_registry.get_tool_count() > 0
    
    await orchestrator.close()

@pytest.mark.asyncio
async def test_execute_task_basic(orchestrator):
    """Test basic task execution"""
    # Mock the client response
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Task completed successfully",
                "tool_calls": []
            }
        }]
    }
    
    with patch.object(orchestrator.client, 'chat_completion', return_value=mock_response):
        with patch.object(orchestrator.completion_detector, 'check_completion', return_value=(True, 0.9, "All done")):
            result = await orchestrator.execute_task(
                task_description="Test task",
                max_iterations=5
            )
            
            assert result["status"] == "completed"
            assert result["task"] == "Test task"
            assert "iterations" in result
    
    await orchestrator.close()

@pytest.mark.asyncio
async def test_tool_execution(orchestrator):
    """Test tool execution"""
    tool_calls = [{
        "id": "test_call_1",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "2 + 2"}'
        }
    }]
    
    results = await orchestrator._execute_tools(tool_calls)
    
    assert len(results) == 1
    assert results[0]["tool_name"] == "calculator"
    assert results[0]["status"] in ["success", "error"]
    
    await orchestrator.close()

def test_can_parallelize(orchestrator):
    """Test parallel execution detection"""
    # Different tools - can parallelize
    tool_calls1 = [
        {"function": {"name": "tool1"}},
        {"function": {"name": "tool2"}}
    ]
    assert orchestrator._can_parallelize(tool_calls1) == True
    
    # Same tool multiple times - cannot parallelize
    tool_calls2 = [
        {"function": {"name": "tool1"}},
        {"function": {"name": "tool1"}}
    ]
    assert orchestrator._can_parallelize(tool_calls2) == False

def test_extract_reasoning(orchestrator):
    """Test reasoning extraction"""
    message1 = {"content": "I did this because it makes sense"}
    reasoning1 = orchestrator._extract_reasoning(message1)
    assert "because" in reasoning1.lower()
    
    message2 = {"content": "Reasoning: This is the explanation"}
    reasoning2 = orchestrator._extract_reasoning(message2)
    assert "explanation" in reasoning2.lower()
