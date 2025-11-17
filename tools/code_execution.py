"""Code execution tool with light sandboxing and package installation support."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional

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
        self.python_executable = sys.executable or "python3"
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Return tool definition for Mistral API
        """
        return {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": (
                    "Execute code in a sandboxed environment. Supports Python, JavaScript, and Bash. "
                    "Optional package installation is available for Python."
                ),
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
                        },
                        "packages": {
                            "type": "array",
                            "description": "Python packages to install before execution",
                            "items": {"type": "string"},
                            "default": []
                        },
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300,
                            "default": 30,
                            "description": "Execution timeout in seconds"
                        }
                    },
                    "required": ["code"]
                }
            }
        }

    async def execute(
        self,
        code: str,
        language: str = "python",
        packages: Optional[Iterable[str]] = None,
        timeout: int = 30
    ) -> str:
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

        timeout = max(1, min(timeout, 300))
        packages_list = [pkg.strip() for pkg in (packages or []) if pkg and pkg.strip()]

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
                install_section = ""
                if language == "python" and packages_list:
                    install_section = self._install_packages(packages_list, timeout)
                    if "[pip-error]" in install_section:
                        return install_section

                command = self._get_command(language, temp_file)
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                )

                formatted = self._format_output(
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    language=language,
                    packages=packages_list,
                    command=command,
                    install_output=install_section,
                )

                return formatted

            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out ({timeout}s limit)"
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
            "python": [self.python_executable, file_path],
            "javascript": ["node", file_path],
            "bash": ["bash", file_path]
        }
        return commands.get(language, ["cat", file_path])

    def _install_packages(self, packages: List[str], timeout: int) -> str:
        """Install Python packages using pip before execution."""
        try:
            result = subprocess.run(
                [self.python_executable, "-m", "pip", "install", *packages],
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except FileNotFoundError as exc:
            logger.error("Python executable not found for pip install: %s", exc)
            return "[pip-error] Python executable not found for package installation"
        except subprocess.TimeoutExpired:
            return f"[pip-error] Package installation timed out ({timeout}s)."

        if result.returncode != 0:
            logger.error("pip install failed: %s", result.stderr.strip())
            return "[pip-error] Package installation failed:\n" + result.stderr

        output = result.stdout.strip()
        if output:
            return "[pip-output]\n" + output
        return ""

    def _format_output(
        self,
        *,
        exit_code: int,
        stdout: str,
        stderr: str,
        language: str,
        packages: List[str],
        command: List[str],
        install_output: str,
    ) -> str:
        """Compose a user-friendly response summarizing execution results."""
        sections: List[str] = []

        sections.append(f"[language] {language}")
        sections.append(f"[command] {' '.join(command)}")
        sections.append(f"[exit-code] {exit_code}")

        if packages:
            sections.append(f"[packages] {', '.join(packages)}")

        if install_output:
            sections.append(install_output.strip())

        stdout_clean = (stdout or "").strip()
        stderr_clean = (stderr or "").strip()

        sections.append("[stdout]\n" + (stdout_clean if stdout_clean else "<empty>"))
        sections.append("[stderr]\n" + (stderr_clean if stderr_clean else "<empty>"))

        if exit_code != 0:
            sections.append("[status] failure")
        else:
            sections.append("[status] success")

        return "\n\n".join(sections)
