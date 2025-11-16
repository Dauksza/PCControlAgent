"""
Vision and multimodal capabilities using Pixtral models
"""
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import io
from mistralai import Mistral
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class VisionManager:
    """
    Handle vision/multimodal tasks with Pixtral models
    - Image analysis
    - Document OCR
    - Visual Q&A
    - Multi-image understanding
    """
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.vision_models = ["pixtral-12b", "pixtral-large"]
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string
        """
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    
    def resize_image(self, image_path: str, max_size: tuple = (1024, 1024)) -> str:
        """
        Resize image for optimal API usage
        """
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded}"
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        model: str = "pixtral-12b",
        resize: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single image with a prompt
        """
        # Encode image
        if resize:
            image_data = self.resize_image(image_path)
        else:
            image_data = self.encode_image_to_base64(image_path)
        
        # Create message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_data}
                ]
            }
        ]
        
        try:
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=2000
            )
            
            return response.model_dump()
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    async def analyze_multiple_images(
        self,
        image_paths: List[str],
        prompt: str,
        model: str = "pixtral-large"
    ) -> Dict[str, Any]:
        """
        Analyze multiple images simultaneously (up to 30)
        """
        if len(image_paths) > 30:
            raise ValueError("Maximum 30 images supported")
        
        # Build content array with text and images
        content = [{"type": "text", "text": prompt}]
        
        for image_path in image_paths:
            image_data = self.resize_image(image_path)
            content.append({"type": "image_url", "image_url": image_data})
        
        messages = [{"role": "user", "content": content}]
        
        try:
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=4000
            )
            
            return response.model_dump()
            
        except Exception as e:
            logger.error(f"Multi-image analysis failed: {e}")
            raise
    
    async def document_ocr(
        self,
        image_path: str,
        extract_tables: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from documents using vision model
        """
        prompt = "Extract all text from this document. "
        if extract_tables:
            prompt += "Preserve table structure using markdown formatting."
        
        return await self.analyze_image(image_path, prompt, model="pixtral-large")
