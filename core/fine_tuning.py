"""
Fine-tuning workflow support for Mistral models
"""
from typing import Dict, List, Optional, Any
import httpx
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class FineTuningManager:
    """
    Manage fine-tuning jobs for custom model training
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = settings.MISTRAL_API_BASE
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=60
        )
    
    async def create_fine_tuning_job(
        self,
        training_file: str,
        model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        validation_file: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new fine-tuning job
        
        Args:
            training_file: Path to training data file
            model: Base model to fine-tune
            hyperparameters: Training hyperparameters
            validation_file: Optional validation data file
            suffix: Optional suffix for the fine-tuned model name
        """
        payload = {
            "model": model,
            "training_file": training_file,
        }
        
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if validation_file:
            payload["validation_file"] = validation_file
        if suffix:
            payload["suffix"] = suffix
        
        try:
            response = await self.http_client.post("/fine_tuning/jobs", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to create fine-tuning job: {e}")
            raise
    
    async def get_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a fine-tuning job
        """
        try:
            response = await self.http_client.get(f"/fine_tuning/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get fine-tuning job: {e}")
            raise
    
    async def list_fine_tuning_jobs(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List all fine-tuning jobs
        """
        try:
            response = await self.http_client.get(
                "/fine_tuning/jobs",
                params={"limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list fine-tuning jobs: {e}")
            raise
    
    async def cancel_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running fine-tuning job
        """
        try:
            response = await self.http_client.post(f"/fine_tuning/jobs/{job_id}/cancel")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to cancel fine-tuning job: {e}")
            raise
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
