"""Tool registry for dynamic tool management."""

import logging
from typing import Any

from nanobot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool execution result as string.

        Raises:
            KeyError: If tool not found.
        """
        logger.info("=" * 80)
        logger.info(f"[INSTRUMENTATION] ToolRegistry.execute() called")
        logger.info(f"[INSTRUMENTATION] Tool name: {name}")
        logger.info(f"[INSTRUMENTATION] Parameters: {params}")
        logger.info(f"[INSTRUMENTATION] Registered tools: {list(self._tools.keys())}")

        tool = self._tools.get(name)
        if not tool:
            logger.error(f"[INSTRUMENTATION] TOOL NOT FOUND: '{name}'")
            logger.error(f"[INSTRUMENTATION] Available tools: {list(self._tools.keys())}")
            return f"Error: Tool '{name}' not found"

        logger.info(f"[INSTRUMENTATION] Tool found: {tool}")
        logger.info(f"[INSTRUMENTATION] Tool class: {tool.__class__.__name__}")

        try:
            logger.info(f"[INSTRUMENTATION] Validating parameters...")
            errors = tool.validate_params(params)
            if errors:
                logger.error(f"[INSTRUMENTATION] Parameter validation FAILED: {errors}")
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)

            logger.info(f"[INSTRUMENTATION] Parameters validated successfully")
            logger.info(f"[INSTRUMENTATION] Executing tool.execute(**params)...")

            result = await tool.execute(**params)

            logger.info(f"[INSTRUMENTATION] Tool execution COMPLETED")
            logger.info(f"[INSTRUMENTATION] Result type: {type(result)}")
            logger.info(f"[INSTRUMENTATION] Result (first 300 chars): {str(result)[:300]}")
            logger.info("=" * 80)

            return result
        except Exception as e:
            logger.error(f"[INSTRUMENTATION] Tool execution FAILED with exception: {e}", exc_info=True)
            logger.error("=" * 80)
            return f"Error executing {name}: {str(e)}"
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
