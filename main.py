import asyncio
import sys
import click

from agent.events import AgentEventType
from client.llm_client import LLMClient
from agent.agent import Agent
from ui.renderer import TUI, get_console

from typing import Any
console = get_console()

class CLI:
    def __init__(self):   
        self.agent: Agent | None = None
        self.tui = TUI(console=console)

    async def run_single(self, message: str )-> str | None:
        async with Agent() as agent:
            # it is instantiated because later we want it in other helper methods  
            self.agent = agent
            return await self._process_message(message)
    
    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool = self.agent.tool_registry.get(tool_name)
        if not tool:
            return None
        kind = getattr(tool, "kind", None)
        return getattr(kind, "value", None)
    
    async def _process_message(self, message: str)-> str | None:
        if not self.agent:
            raise ValueError("Agent not initialized.")

        assistant_streaming = False
        final_response : str | None = None
        
        async for event in self.agent.run(message=message):
            # print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error")
                console.print(f"[error]Agent Error:[/error] {error}")
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("tool_name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                
                self.tui.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )

        return final_response

async def run(messages: dict[str,Any]):
    client = LLMClient()
    async for event in client.chat_completion(messages=messages, stream=True):
        print(event)
    # await client.close()

@click.command()
@click.argument("prompt", required=False)

def main(
    prompt: str | None,
):
    cli = CLI()
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)

if __name__ == "__main__":
    main()