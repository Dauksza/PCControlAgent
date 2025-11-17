"""
Task decomposition for breaking down complex tasks
"""
from typing import List, Dict, Any, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)

class TaskDecomposer:
    """
    Decompose complex tasks into manageable subtasks
    """
    
    def __init__(self):
        self.decomposition_patterns = {
            "search_and_analyze": [
                "Search for relevant information",
                "Analyze and summarize findings",
                "Provide structured response"
            ],
            "multi_step_code": [
                "Understand requirements",
                "Write code implementation",
                "Test and verify code",
                "Document the solution"
            ],
            "research": [
                "Gather information from multiple sources",
                "Compare and contrast findings",
                "Synthesize conclusions",
                "Present final report"
            ]
        }
    
    async def decompose(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a task into subtasks
        
        Args:
            task_description: Description of the main task
        
        Returns:
            List of subtask dictionaries
        """
        # Simple heuristic-based decomposition
        # In production, this could use an LLM for better decomposition
        
        subtasks = []
        task_lower = task_description.lower()
        
        # Detect task patterns
        if "search" in task_lower and ("analyze" in task_lower or "summarize" in task_lower):
            pattern = self.decomposition_patterns["search_and_analyze"]
        elif "code" in task_lower or "implement" in task_lower or "write" in task_lower:
            pattern = self.decomposition_patterns["multi_step_code"]
        elif "research" in task_lower or "compare" in task_lower:
            pattern = self.decomposition_patterns["research"]
        else:
            # Default generic decomposition
            pattern = [
                "Understand the task requirements",
                "Execute the task",
                "Verify and provide results"
            ]
        
        for i, step in enumerate(pattern, 1):
            subtasks.append({
                "id": i,
                "description": step,
                "status": "pending",
                "started_at": None,
                "completed_at": None
            })
        
        logger.info(f"Decomposed task into {len(subtasks)} subtasks")
        return subtasks
    
    def update_subtask_status(
        self,
        subtasks: List[Dict[str, Any]],
        subtask_id: int,
        status: str
    ):
        """
        Update the status of a subtask
        
        Args:
            subtasks: List of subtasks
            subtask_id: ID of subtask to update
            status: New status (pending, in_progress, completed, failed)
        """
        for subtask in subtasks:
            if subtask["id"] == subtask_id:
                subtask["status"] = status
                logger.info(f"Subtask {subtask_id} status updated to: {status}")
                break
    
    def get_next_pending_subtask(
        self,
        subtasks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next pending subtask
        
        Returns:
            Next pending subtask or None
        """
        for subtask in subtasks:
            if subtask["status"] == "pending":
                return subtask
        return None
    
    def get_completion_percentage(self, subtasks: List[Dict[str, Any]]) -> float:
        """
        Calculate task completion percentage
        
        Returns:
            Completion percentage (0.0 to 1.0)
        """
        if not subtasks:
            return 0.0
        
        completed = sum(1 for s in subtasks if s["status"] == "completed")
        return completed / len(subtasks)
