"""Automation tool powered by Playwright for browser interactions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_tool import BaseTool
from utils.logging_config import get_logger

logger = get_logger(__name__)

try:  # Playwright may not be installed in every environment
    from playwright.async_api import Browser, Page, async_playwright
except ImportError:  # pragma: no cover - optional dependency
    Browser = Page = None  # type: ignore
    async_playwright = None  # type: ignore


class PlaywrightAutomationTool(BaseTool):
    """High-level wrapper around Playwright for browser control."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "playwright_automation"
        self.description = "Automate browser actions using Playwright."
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
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
                                "navigate",
                                "click",
                                "type_text",
                                "screenshot",
                                "get_text",
                            ],
                        },
                        "url": {"type": "string"},
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "wait": {
                            "type": "integer",
                            "description": "Optional wait time in milliseconds",
                            "default": 0,
                        },
                    },
                    "required": ["operation"],
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        if async_playwright is None:
            return "Playwright is not installed. Run 'pip install playwright' and install browsers." \
                " See https://playwright.dev/python/docs/intro"

        operation = kwargs.get("operation")
        if not operation:
            return "Missing operation."

        handler = getattr(self, f"_{operation}", None)
        if not handler:
            return f"Unsupported operation: {operation}"

        try:
            return await handler(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("Playwright operation %s failed: %s", operation, exc)
            return f"Playwright error: {exc}"

    async def _ensure_page(self) -> Page:
        if self._page:
            return self._page

        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._browser is None:
            self._browser = await self._playwright.chromium.launch(headless=True)
        context = await self._browser.new_context()
        self._page = await context.new_page()
        return self._page

    async def _navigate(self, url: str, **_: Any) -> str:
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url}"

    async def _click(self, selector: str, wait: int = 0, **_: Any) -> str:
        page = await self._ensure_page()
        await page.click(selector, timeout=wait or 30000)
        return f"Clicked selector: {selector}"

    async def _type_text(self, selector: str, text: str, wait: int = 0, **_: Any) -> str:
        page = await self._ensure_page()
        await page.fill(selector, text, timeout=wait or 30000)
        return f"Typed text into {selector}"

    async def _screenshot(self, **_: Any) -> str:
        page = await self._ensure_page()
        file_path = self._screenshot_dir / "playwright_screenshot.png"
        counter = 1
        while file_path.exists():
            file_path = self._screenshot_dir / f"playwright_screenshot_{counter}.png"
            counter += 1
        await page.screenshot(path=str(file_path), full_page=True)
        return f"Screenshot saved to {file_path}"

    async def _get_text(self, selector: str, wait: int = 0, **_: Any) -> str:
        page = await self._ensure_page()
        await page.wait_for_selector(selector, timeout=wait or 30000)
        text = await page.inner_text(selector)
        return text or ""

    async def close(self) -> None:
        if self._page:
            await self._page.context.close()
            self._page = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        if self._playwright:
            try:
                loop = getattr(self._playwright, "loop", None)
                if loop and loop.is_running():
                    loop.create_task(self.close())
            except Exception:
                pass
