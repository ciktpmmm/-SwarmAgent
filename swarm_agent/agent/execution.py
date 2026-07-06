from __future__ import annotations

from .base import Agent, AgentConfig
from swarm_agent.llm import LLM, Message


class ExecutionAgent(Agent):
    """执行智能体 — 仅作为结构容器，所有行为由 Spawner Agent 通过 AgentSpec 定义。"""

    def __init__(self, llm: LLM, config: AgentConfig, tool_map=None, verbose=True):
        super().__init__(llm, config, tool_map, verbose)

    async def run(self, user_input: str) -> str:
        messages = [Message(role="system", content=self._config.system_prompt)]
        messages.append(Message(role="user", content=user_input))

        response = await self._llm.chat(
            messages,
            tools=self.tool_specs or None,
            temperature=self._config.temperature,
        )

        return response.content or ""
