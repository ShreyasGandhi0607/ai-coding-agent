import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Any
from typing import AsyncGenerator
from client.response import StreamEventType, TextDelta, TokenUsage, StreamEvent
from openai import RateLimitError,APIConnectionError,APIError
import asyncio

load_dotenv()

class LLMClient:
    def __init__(self) ->None:
        self._client: AsyncOpenAI | None = None
        self._max_retries : int = 3

    # Singleton pattern to ensure only one client instance
    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        return self._client
    
    # Gracefully close the client session
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
    
    def _build_tools(self, tools: list[dict[str,Any]]):
        return [
            {
                'type': 'function',
                'function' : {
                    'name': tool['name'],
                    'description': tool.get('description',''),
                    'parameters': tool.get(
                        "paramters",
                        {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }
                    )
                }
            } for tool in tools
        ]

    async def chat_completion(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            stream: bool = True,
            )-> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
                    "model": "mistralai/devstral-2512:free",
                    "messages": messages,
                    "stream": stream,

                }
        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries+1):
            try:
                if stream:
                    async for event in self._stream_response(client=client, kwargs=kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client=client, kwargs=kwargs)
                    yield event  # yield the single event for non-streaming response
                    #Note : differnce between yield and return is that yield allows the function to be a generator, producing a series of values over time, whereas return exits the function and provides a single value.
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    # Exponential backoff before retrying
                    # attemmpt -> failed
                    #  1s -> failed
                    #  2s -> failed
                    #  4s -> failed
                    #  exponential backoff = 2 ** attempt
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded after {self._max_retries} retries: {str(e)}"
                    )
                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API connection error after {self._max_retries} retries: {str(e)}"
                    )
                    return
            except APIError as e:
                yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API connection error after {self._max_retries} retries: {str(e)}"
                    )
                return
        
    async def _stream_response(
            self,
            client : AsyncOpenAI,
            kwargs: dict[str, Any]
            )-> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)
        
        finish_reason : str | None = None
        usage : TokenUsage | None = None
        
        async for chunk in response:
            if hasattr(chunk,"usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content=delta.content),
                )
            
            print(delta.tool_calls)
        
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
            self,
            client : AsyncOpenAI,
            kwargs: dict[str, Any]
            ):
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)
        
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
        
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )