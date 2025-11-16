"""
Integration tests for full workflow
"""
import pytest
from unittest.mock import patch, Mock
from agent.orchestrator import AgentOrchestrator

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_workflow_with_mocked_api():
    """Test complete workflow with mocked API responses"""
    orchestrator = AgentOrchestrator(api_key="test_key")
    
    # Mock responses for multiple iterations
    mock_responses = [
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I will search for information",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test query", "num_results": 3}'
                        }
                    }]
                }
            }]
        },
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Based on the search results, here is the summary. Task completed.",
                    "tool_calls": []
                }
            }]
        }
    ]
    
    call_count = [0]
    
    async def mock_chat_completion(*args, **kwargs):
        response = mock_responses[min(call_count[0], len(mock_responses) - 1)]
        call_count[0] += 1
        return response
    
    with patch.object(orchestrator.client, 'chat_completion', side_effect=mock_chat_completion):
        with patch.object(orchestrator.tool_registry, 'execute_tool', return_value="Mocked search results"):
            result = await orchestrator.execute_task(
                task_description="Search for test information and summarize",
                max_iterations=5
            )
            
            assert result is not None
            assert "iterations" in result
            assert len(result["iterations"]) >= 1
            assert "tool_calls" in result
    
    await orchestrator.close()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_workflow_with_multiple_tools():
    """Test workflow that uses multiple tools"""
    orchestrator = AgentOrchestrator(api_key="test_key")
    
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Executing tools",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "10 * 5"}'
                        }
                    },
                    {
                        "id": "call_2",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "100 / 2"}'
                        }
                    }
                ]
            }
        }]
    }
    
    tool_calls = mock_response["choices"][0]["message"]["tool_calls"]
    results = await orchestrator._execute_tools(tool_calls)
    
    assert len(results) == 2
    assert all(r["tool_name"] == "calculator" for r in results)
    
    await orchestrator.close()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_max_iterations_handling():
    """Test that orchestrator respects max iterations"""
    orchestrator = AgentOrchestrator(api_key="test_key")
    
    # Mock response that never completes
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Still working on it...",
                "tool_calls": []
            }
        }]
    }
    
    with patch.object(orchestrator.client, 'chat_completion', return_value=mock_response):
        with patch.object(orchestrator.completion_detector, 'check_completion', return_value=(False, 0.3, "Not done")):
            result = await orchestrator.execute_task(
                task_description="Impossible task",
                max_iterations=3
            )
            
            assert result["status"] == "max_iterations_reached"
            assert result["total_iterations"] == 3
    
    await orchestrator.close()
