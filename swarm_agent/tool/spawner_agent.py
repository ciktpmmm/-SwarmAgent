from __future__ import annotations

import json

from swarm_agent.agent.base import AgentSpec
from swarm_agent.llm import LLM, LLMError, Message
from swarm_agent.tool import Tool

from .ecosystem_registration import EcosystemRegistration

_SPAWNER_SYSTEM_PROMPT = """你是群智能体框架的核心——Spawner Agent。

你的职责是根据用户的需求，分析并规划出一个最合适的执行智能体。
你需要输出严格的 JSON 格式，包含以下字段：

{
    "name": "智能体名称，英文，如 code_reviewer",
    "role": "智能体的角色描述，一句话说明",
    "system_prompt": "给执行智能体的系统提示词，详细描述其职责、行为规范、输出要求",
    "tools": ["工具名数组，从可用工具列表中选取最相关的"],
    "max_iterations": 15,
    "temperature": 0.7
}

分析要求：
1. 理解用户需求的本质任务类型（信息检索、代码编写、数据分析、文档处理等）
2. 为执行智能体赋予明确的角色定位
3. 撰写详细、可操作的 system_prompt
4. 从可用工具列表中精选最匹配的工具
5. 根据任务复杂度合理设置 max_iterations"""


class SpawnerAgent:
    """Spawner Agent - 群智能体框架的核心。

    根据用户需求动态分析、规划并创建执行智能体。
    """

    def __init__(
        self,
        llm: LLM,
        registry: EcosystemRegistration | None = None,
        verbose: bool = True,
    ):
        self._llm = llm
        self._registry = registry or EcosystemRegistration()
        self._verbose = verbose
        self._created_agents: dict[str, AgentSpec] = {}

    @property
    def registry(self) -> EcosystemRegistration:
        return self._registry

    @property
    def created_agents(self) -> dict[str, AgentSpec]:
        return dict(self._created_agents)

    def register_tool(self, tool: Tool) -> None:
        self._registry.register_tool(tool)

    def register_tools(self, *tools: Tool) -> None:
        self._registry.register_tools(*tools)

    async def create_agent(self, requirement: str) -> AgentSpec:
        """分析用户需求，生成执行智能体的规格说明。"""
        tool_hint = f"\n可用工具: {self._registry.available_tool_names}"
        messages = [
            Message(role="system", content=_SPAWNER_SYSTEM_PROMPT + tool_hint),
            Message(role="user", content=f"需求: {requirement}"),
        ]

        response = await self._llm.chat(messages, temperature=0.3, max_tokens=2000)

        if not response.content:
            raise LLMError("Spawner Agent 未能生成智能体规格")

        spec_dict = self._parse_response(response.content)
        spec = AgentSpec(
            name=spec_dict["name"],
            role=spec_dict["role"],
            system_prompt=spec_dict["system_prompt"],
            tools=spec_dict.get("tools", []),
            max_iterations=spec_dict.get("max_iterations", 15),
            temperature=spec_dict.get("temperature", 0.7),
        )

        self._created_agents[spec.name] = spec

        if self._verbose:
            print(f"[Spawner] 已创建智能体: {spec.name} ({spec.role})")
            print(f"[Spawner] 分配工具: {spec.tools}")

        return spec

    async def spawn(self, requirement: str) -> AgentSpec:
        return await self.create_agent(requirement)

    async def create_and_run(self, requirement: str, task: str) -> str:
        spec = await self.create_agent(requirement)
        agent = self._registry.build_agent(self._llm, spec)
        return await agent.run(task)

    def _parse_response(self, content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("\n", 1)[0]
            if content.endswith("```"):
                content = content[:-3]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start:end + 1])
            raise LLMError(f"无法解析 Spawner Agent 输出: {content[:200]}")
