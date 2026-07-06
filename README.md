# 群智能体 (Swarm Agent)

**Spawner Agent 驱动的动态智能体创建框架。**

一个多智能体生态框架——Spawner Agent 在运行时根据生态需求"孕育"新的执行智能体。**
如同自然生态系统中的物种分化，群智能体不预定义所有智能体角色，
而是在运行时由 Spawner Agent 感知需求、生成 AgentSpec，动态"孕育"出
适配特定任务的执行智能体，形成持续演化的智能体生态。

## 快速开始

## 项目结构

```
swarm_agent/
├── spawner/                        # Spawner Agent 模块
│   ├── spawner_agent.py            # 创建智能体 → AgentSpec
│   └── ecosystem_registration.py   # 工具注册 + 智能体实例化
├── agent/
│   ├── base.py                     # Agent 基类 + AgentSpec
│   └── execution.py                # 执行智能体容器
├── llm/                            # LLM 接口 + 适配器
└── tool/                           # Tool 基类 + 内置工具
```
