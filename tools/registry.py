# manage entire dictionary of tools
# register new tools here

import logging
from pathlib import Path
from typing import Any
from tools.base import Tool, ToolInvocation, ToolResult
from tools.builtin import ReadFileTool, get_all_builtin_tools

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool : {tool.name}")
        
        self._tools[tool.name] = tool   
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str)-> bool:
        if name in self._tools:
            del self._tools[name]
            return True
    
        return False
    
    def get(self, name: str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]
        return None
    
    def get_tools(self) -> list[Tool]:
        tools : list[Tool] = []

        for tool in self._tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self) -> list[dict[str,Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]
    
    async def invoke(self, name: str, params: dict[str,Any] | None, cwd: Path,)->ToolResult:
        tool = self.get(name)

        if tool is None:
            return ToolResult.error_result(
                f"Unknown tool: {name}",
                metadata={"tool_name": name},
            )
        
        invocation_params: dict[str, Any] = params or {}

        validation_errors = tool.validate_params(invocation_params)

        if validation_errors:
            return ToolResult.error_result(
                f"Invalid parameters for tool {name}: " + "; ".join(validation_errors),
                metadata={"tool_name": name, "validation_errors": validation_errors},
            )
        invocation = ToolInvocation(
            params=invocation_params,
            cwd=cwd,
        )
        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Error executing tool {name}: {str(e)}")
            result = ToolResult.error_result(
                f"Internal Error executing tool {name}: {str(e)}",
                metadata={"tool_name": name},
            )

        return result

def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    # register default tools here
    for tool_cls in get_all_builtin_tools():
        registry.register(tool_cls())
    return registry