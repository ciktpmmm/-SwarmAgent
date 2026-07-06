from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from swarm_agent.llm import LLM, Message
from swarm_agent.tool import Tool, ToolResult


@dataclass
class AgentConfig:
    name: str = ""
    role: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 15
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentError(Exception):
    pass


class Agent(ABC):

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        tool_map: dict[str, Tool] | None = None,
        verbose: bool = True,
    ):
        self._llm = llm
        self._config = config
        self._tool_map = tool_map or {}
        self._verbose = verbose

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def tool_specs(self) -> list[dict]:
        return [self._tool_map[name].to_openai_spec() for name in self._config.tools if name in self._tool_map]

    async def _run_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tool_map.get(name)
        if not tool:
            return ToolResult(success=False, output=f"未知工具: {name}")
        return await tool.run(**args)

    @abstractmethod
    async def run(self, user_input: str) -> str:
        ...


@dataclass
class AgentSpec:
    name: str
    role: str
    system_prompt: str
    tools: list[str]
    max_iterations: int = 15
    temperature: float = 0.7
