from __future__ import annotations

from swarm_agent.agent import AgentConfig, ExecutionAgent
from swarm_agent.agent.base import AgentSpec
from swarm_agent.llm import LLM
from swarm_agent.tool import Tool, ToolError


class EcosystemRegistration:
    """工具注册中心 + AgentSpec 校验 + 执行智能体实例化。"""

    def __init__(self, tool_registry: dict[str, Tool] | None = None):
        self._tool_registry = tool_registry or {}

    def register_tool(self, tool: Tool) -> None:
        self._tool_registry[tool.name] = tool

    def register_tools(self, *tools: Tool) -> None:
        for tool in tools:
            self.register_tool(tool)

    @property
    def available_tools(self) -> list[Tool]:
        return list(self._tool_registry.values())

    @property
    def available_tool_names(self) -> list[str]:
        return list(self._tool_registry.keys())

    def validate_spec(self, spec: AgentSpec) -> AgentSpec:
        invalid = [t for t in spec.tools if t not in self._tool_registry]
        if invalid:
            raise ToolError(f"未知工具: {invalid}，可用工具: {self.available_tool_names}")
        return spec

    def build_agent(self, llm: LLM, spec: AgentSpec) -> ExecutionAgent:
        self.validate_spec(spec)
        config = AgentConfig(
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt,
            tools=spec.tools,
            max_iterations=spec.max_iterations,
            temperature=spec.temperature,
        )
        return ExecutionAgent(llm=llm, config=config, tool_map=self._tool_registry)
