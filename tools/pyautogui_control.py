"""Desktop automation tool leveraging PyAutoGUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import pyautogui

    pyautogui.FAILSAFE = False
except Exception as exc:  # pragma: no cover - dependency might be missing or display unavailable
    pyautogui = None  # type: ignore
    logger.warning("PyAutoGUI unavailable: %s", exc)


class PyAutoGUIControlTool(BaseTool):
    """Provide guarded access to PyAutoGUI desktop automation primitives."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "pyautogui_control"
        self.description = "Control the desktop using PyAutoGUI with explicit confirmations."
        self._screenshot_dir = Path("exports") / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

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
                                "move_mouse",
                                "click_at",
                                "type_keys",
                                "press_key",
                                "take_screenshot",
                                "get_window_list",
                            ],
                        },
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "text": {"type": "string"},
                        "key": {"type": "string"},
                        "button": {
                            "type": "string",
                            "enum": ["left", "right", "middle"],
                            "default": "left",
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Must be true for operations that modify the desktop",
                            "default": False,
                        },
                    },
                    "required": ["operation"],
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        if pyautogui is None:
            return "PyAutoGUI is not available in this environment. Install it with 'pip install pyautogui'."

        operation = kwargs.get("operation")
        if not operation:
            return "Missing operation."

        handler = getattr(self, f"_{operation}", None)
        if not handler:
            return f"Unsupported operation: {operation}"

        try:
            return handler(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("PyAutoGUI operation %s failed: %s", operation, exc)
            return f"PyAutoGUI error: {exc}"

    def _require_confirmation(self, confirm: bool) -> None:
        if not confirm:
            raise PermissionError("Operation requires explicit confirmation (confirm=true)")

    def _move_mouse(self, x: int, y: int, confirm: bool = False, **_: Any) -> str:
        self._require_confirmation(confirm)
        pyautogui.moveTo(x, y, duration=0.25)
        return f"Mouse moved to ({x}, {y})"

    def _click_at(self, x: int, y: int, button: str = "left", confirm: bool = False, **_: Any) -> str:
        self._require_confirmation(confirm)
        pyautogui.click(x=x, y=y, button=button)
        return f"Clicked {button} button at ({x}, {y})"

    def _type_keys(self, text: str, confirm: bool = False, **_: Any) -> str:
        self._require_confirmation(confirm)
        pyautogui.write(text, interval=0.02)
        return f"Typed text ({len(text)} chars)"

    def _press_key(self, key: str, confirm: bool = False, **_: Any) -> str:
        self._require_confirmation(confirm)
        pyautogui.press(key)
        return f"Pressed key: {key}"

    def _take_screenshot(self, **_: Any) -> str:
        image = pyautogui.screenshot()
        file_path = self._screenshot_dir / "pyautogui_screenshot.png"
        counter = 1
        while file_path.exists():
            file_path = self._screenshot_dir / f"pyautogui_screenshot_{counter}.png"
            counter += 1
        image.save(file_path)
        return f"Screenshot saved to {file_path}"

    def _get_window_list(self, **_: Any) -> str:
        try:
            titles: List[str] = [title for title in pyautogui.getAllTitles() if title.strip()]  # type: ignore[attr-defined]
        except Exception:
            return "Window enumeration not supported on this platform."
        return "\n".join(titles) if titles else "<no windows detected>"
