from __future__ import annotations

from typing import Any

from .base import LLM, LLMError, LLMResponse, Message, ToolCall

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAILLM(LLM):

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if AsyncOpenAI is None:
            raise LLMError("openai package not installed. pip install swarm-agent[openai]")
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(messages, tools, temperature, max_tokens)
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}") from e

        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content,
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, args=tc.function.arguments)
                for tc in choice.message.tool_calls or []
            ],
            usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            if resp.usage
            else None,
        )

    async def chat_stream(self, messages, tools=None, temperature=None, max_tokens=None):
        kwargs = self._build_kwargs(messages, tools, temperature, max_tokens)
        kwargs["stream"] = True
        async for chunk in await self._client.chat.completions.create(**kwargs):
            yield chunk

    def _build_kwargs(self, messages, tools, temperature, max_tokens) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs
