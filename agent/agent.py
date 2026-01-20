from __future__ import annotations
from typing import Any, AsyncGenerator
from client.llm_client import LLMClient
from agent.events import AgentEvent, AgentEventType
from client.response import StreamEventType, ToolCall, ToolResultMessage
from context.manager import ContextManager
from tools.registry import create_default_registry
from pathlib import Path
# this entire class just processes one single message and runs one single time for one message
class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def run(self, message : str):
        yield AgentEvent.agent_start(message=message)
        self.context_manager.add_user_message(message)
        #  add user message to context
        final_response : str | None = None
        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")  

        yield AgentEvent.agent_end(response=final_response)

    async def _agentic_loop(self)-> AsyncGenerator[AgentEvent, None]:
        # messages = [{"role": "user", "content": "Hey what is going on."}]
        response_text = ""
        
        tool_schemas = self.tool_registry.get_schemas()

        tool_calls : list[ToolCall] = []

        async for event in self.client.chat_completion(
            messages=self.context_manager.get_messages(),
            tools=tool_schemas if tool_schemas else None,
            stream=True,
        ):
            # print(event)
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:    
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content=content)

            elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                if event.tool_call:
                    tool_calls.append(event.tool_call)


            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(
                    error=event.error or "Unknown error occured",
                    details={},
                )    
        self.context_manager.add_assistant_message(response_text or None,)

        if response_text:
            yield AgentEvent.text_complete(content=response_text)
        tool_call_results : list[ToolResultMessage] = []
        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(
                tool_call.call_id,
                tool_call.name,
                tool_call.arguments,
            )
            result = await self.tool_registry.invoke(
                tool_call.name,
                tool_call.arguments,
                Path.cwd(),
            )

            yield AgentEvent.tool_call_complete(
                tool_call.call_id,
                tool_call.name,
                result,
            )

            tool_call_results.append(
                ToolResultMessage(
                    tool_call_id=tool_call.call_id,
                    content=result.to_model_output(),
                    is_error=not result.success,
                )
            )
    
        for tool_result in tool_call_results:
            self.context_manager.add_tool_result_message(
                tool_result.tool_call_id,
                tool_result.content,
            )

    async def __aenter__(self)->Agent:
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback)->None:
        if self.client:
            await self.client.close()
            self.client = None