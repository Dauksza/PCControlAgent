"""
Code execution tool (sandboxed)
"""
from typing import Dict, Any
import subprocess
import tempfile
import os
from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)

class CodeExecutionTool(BaseTool):
    """
    Execute code in a sandboxed environment
    WARNING: This is a simplified implementation. Production use requires proper sandboxing.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "code_interpreter"
        self.description = "Execute Python code in a sandboxed environment"
        self.supported_languages = ["python", "javascript", "bash"]
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Return tool definition for Mistral API
        """
        return {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute code in a sandboxed environment. Supports Python, JavaScript, and Bash.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "enum": ["python", "javascript", "bash"],
                            "default": "python"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    
    async def execute(self, code: str, language: str = "python") -> str:
        """
        Execute code
        
        Args:
            code: Code to execute
            language: Programming language
        
        Returns:
            Execution output
        """
        if language not in self.supported_languages:
            return f"Unsupported language: {language}. Supported: {', '.join(self.supported_languages)}"
        
        try:
            logger.info(f"Executing {language} code")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=self._get_file_extension(language),
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute code
                result = subprocess.run(
                    self._get_command(language, temp_file),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False
                )
                
                output = result.stdout
                errors = result.stderr
                
                if result.returncode != 0:
                    return f"Execution failed:\n{errors}"
                
                return f"Output:\n{output}" if output else "Execution completed successfully (no output)"
                
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10s limit)"
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Execution error: {str(e)}"
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh"
        }
        return extensions.get(language, ".txt")
    
    def _get_command(self, language: str, file_path: str) -> list:
        """Get execution command for language"""
        commands = {
            "python": ["python3", file_path],
            "javascript": ["node", file_path],
            "bash": ["bash", file_path]
        }
        return commands.get(language, ["cat", file_path])
