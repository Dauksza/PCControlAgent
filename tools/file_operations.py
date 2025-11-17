"""Tool for safe file system interactions within the project workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FileOperationsTool(BaseTool):
    """Provide read/write helpers with guardrails for destructive actions."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "file_operations"
        self.description = "Perform controlled file system operations (read, write, list, delete, edit)."

    def get_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "read_file",
                                "write_file",
                                "list_files",
                                "delete_file",
                                "edit_file",
                            ],
                            "description": "Requested file operation",
                        },
                        "path": {
                            "type": "string",
                            "description": "Target file or directory path relative to workspace",
                        },
                        "content": {
                            "type": "string",
                            "description": "New file content (write_file)",
                        },
                        "changes": {
                            "type": "array",
                            "description": "List of find/replace instructions (edit_file)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "find": {"type": "string"},
                                    "replace": {"type": "string"},
                                },
                                "required": ["find", "replace"],
                            },
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag for destructive actions",
                            "default": False,
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to list directories recursively",
                            "default": False,
                        },
                    },
                    "required": ["operation", "path"],
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        operation = kwargs.get("operation")
        path = kwargs.get("path")
        if not operation or not path:
            return "Invalid request: operation and path are required."

        method = getattr(self, operation, None)
        if not callable(method):
            return f"Unsupported operation: {operation}"

        try:
            result = method(Path(path), **{k: v for k, v in kwargs.items() if k not in {"operation", "path"}})
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            logger.error("File operation %s failed: %s", operation, exc)
            return f"Operation failed: {exc}"

    def _ensure_within_workspace(self, path: Path) -> Path:
        """Prevent escaping the repository root."""
        base = Path.cwd().resolve()
        target = (base / path).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError("Path escapes workspace boundary")
        return target

    def read_file(self, path: Path, **_: Any) -> str:
        target = self._ensure_within_workspace(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: Path, content: Optional[str] = None, **_: Any) -> str:
        target = self._ensure_within_workspace(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content or ""
        target.write_text(data, encoding="utf-8")
        logger.info("Wrote file: %s", target)
        return f"File written: {path} (bytes={len(data.encode('utf-8'))})"

    def list_files(self, path: Path, recursive: bool = False, **_: Any) -> str:
        target = self._ensure_within_workspace(path)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: List[str] = []
        if recursive:
            for child in sorted(target.rglob("*")):
                entries.append(str(child.relative_to(Path.cwd())))
        else:
            for child in sorted(target.iterdir()):
                entries.append(str(child.relative_to(Path.cwd())))

        return "\n".join(entries) if entries else "<empty>"

    def delete_file(self, path: Path, confirm: bool = False, **_: Any) -> str:
        if not confirm:
            return "Deletion aborted: confirmation required."
        target = self._ensure_within_workspace(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if target.is_dir():
            raise IsADirectoryError("Refusing to delete directories with this tool")
        target.unlink()
        logger.info("Deleted file: %s", target)
        return f"File deleted: {path}"

    def edit_file(self, path: Path, changes: Optional[List[Dict[str, str]]] = None, **_: Any) -> str:
        if not changes:
            raise ValueError("No changes provided for edit operation")
        target = self._ensure_within_workspace(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = target.read_text(encoding="utf-8")
        replacements = 0
        for change in changes:
            find = change.get("find")
            replace = change.get("replace")
            if find is None or replace is None:
                continue
            if find in content:
                content = content.replace(find, replace)
                replacements += 1

        target.write_text(content, encoding="utf-8")
        logger.info("Edited file %s (changes=%d)", target, replacements)
        return f"File edited: {path} (changes_applied={replacements})"
