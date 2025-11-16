"""
Completion detection logic to ensure tasks are fully done
"""
from typing import List, Dict, Any, Tuple
from utils.logging_config import get_logger

logger = get_logger(__name__)

class CompletionDetector:
    """
    Detect when a task is truly complete with high confidence
    Implements "never-stop-short" logic
    """
    
    def __init__(self):
        self.completion_keywords = [
            "complete", "finished", "done", "accomplished",
            "successfully", "final", "concluded"
        ]
        self.incompletion_keywords = [
            "pending", "todo", "remaining", "need to",
            "should", "will", "next", "incomplete"
        ]
    
    async def check_completion(
        self,
        task_description: str,
        conversation_history: List[Dict[str, Any]],
        subtasks: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]]
    ) -> Tuple[bool, float, str]:
        """
        Check if task is complete
        
        Args:
            task_description: Original task description
            conversation_history: Full conversation history
            subtasks: List of subtasks with their statuses
            tool_results: Results from tool executions
        
        Returns:
            Tuple of (is_complete, confidence, reasoning)
        """
        confidence = 0.0
        reasons = []
        
        # Check 1: All subtasks completed
        if subtasks:
            completed_subtasks = sum(1 for s in subtasks if s["status"] == "completed")
            subtask_ratio = completed_subtasks / len(subtasks)
            
            if subtask_ratio == 1.0:
                confidence += 0.4
                reasons.append("All subtasks completed")
            elif subtask_ratio >= 0.8:
                confidence += 0.2
                reasons.append(f"{int(subtask_ratio * 100)}% of subtasks completed")
            else:
                reasons.append(f"Only {int(subtask_ratio * 100)}% of subtasks completed")
        
        # Check 2: Recent messages indicate completion
        if conversation_history:
            last_messages = conversation_history[-3:]  # Check last 3 messages
            completion_signals = 0
            incompletion_signals = 0
            
            for msg in last_messages:
                content = str(msg.get("content", "")).lower()
                
                for keyword in self.completion_keywords:
                    if keyword in content:
                        completion_signals += 1
                        break
                
                for keyword in self.incompletion_keywords:
                    if keyword in content:
                        incompletion_signals += 1
                        break
            
            if completion_signals > incompletion_signals:
                confidence += 0.3
                reasons.append("Recent messages indicate completion")
            elif incompletion_signals > 0:
                confidence -= 0.2
                reasons.append("Recent messages indicate incomplete work")
        
        # Check 3: Tool execution success
        if tool_results:
            successful_tools = sum(1 for r in tool_results if r.get("status") == "success")
            tool_success_ratio = successful_tools / len(tool_results)
            
            if tool_success_ratio >= 0.8:
                confidence += 0.2
                reasons.append(f"Tool execution success rate: {int(tool_success_ratio * 100)}%")
            else:
                reasons.append(f"Tool execution issues: {int(tool_success_ratio * 100)}% success rate")
        
        # Check 4: No pending tool calls in last message
        if conversation_history:
            last_msg = conversation_history[-1]
            if last_msg.get("role") == "assistant" and not last_msg.get("tool_calls"):
                confidence += 0.1
                reasons.append("No pending tool calls")
        
        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))
        
        is_complete = confidence > 0.7
        reasoning = "; ".join(reasons)
        
        logger.info(f"Completion check: {is_complete} (confidence: {confidence:.2f})")
        logger.info(f"Reasoning: {reasoning}")
        
        return is_complete, confidence, reasoning
    
    def requires_verification(self, confidence: float) -> bool:
        """
        Check if additional verification is needed
        
        Args:
            confidence: Completion confidence score
        
        Returns:
            True if verification needed
        """
        return 0.6 <= confidence <= 0.85
