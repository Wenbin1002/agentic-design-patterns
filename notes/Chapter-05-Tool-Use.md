# 第 5 章：工具使用（Tool Use）

> **一句话理解**：让 Agent 从"只会说话"变成"能做事" — 通过函数调用机制连接外部 API、数据库、代码执行器，打破 LLM 训练数据的限制。

---

## 为什么需要它？

前四章的 Agent 本质上还是在**文本世界里打转** — 生成文本、评估文本、路由文本。但 LLM 有一个根本问题：

- 知识是**静态的**（截止到训练时间）
- 无法**执行操作**（不能发邮件、不能下单）
- 无法**获取实时信息**（不知道今天天气）
- 无法**精确计算**（大数乘法经常算错）

工具使用是把 LLM 从"闭卷考试"变成"开卷考试 + 可以用计算器"的关键一步。

---

## 核心机制：6 步工作流

```
用户: "伦敦天气怎么样？"
    ↓
① [工具定义] → LLM 知道有哪些工具可用（名称、用途、参数）
    ↓
② [LLM 决策] → "我需要调用天气工具"
    ↓
③ [生成调用] → {"tool": "get_weather", "params": {"location": "London"}}
    ↓
④ [框架执行] → 实际调用天气 API（LLM 不执行，框架执行）
    ↓
⑤ [返回结果] → "多云，15°C"
    ↓
⑥ [LLM 整合] → "伦敦目前多云，温度 15°C，建议带件外套。"
```

**关键点**：LLM 只负责**决定用什么工具 + 生成参数**，**实际执行由框架/编排层完成**。LLM 本身不会真的调用 API。

---

## 6 大工具类别

| 类别 | 工具示例 | 解决的问题 |
|------|----------|-----------|
| **信息检索** | 天气 API、搜索引擎、知识库 | 获取实时/外部信息 |
| **数据库/API** | 库存查询、订单状态、支付处理 | 读写业务数据 |
| **计算分析** | 计算器、股票 API、电子表格 | 精确数学计算 |
| **通信** | 邮件 API、消息推送 | 执行发送操作 |
| **代码执行** | Python 解释器、沙盒环境 | 运行代码验证逻辑 |
| **设备控制** | 智能家居 API、IoT 平台 | 控制物理世界 |

---

## 代码示例要点

### LangChain 方式 — `@tool` 装饰器

```python
@langchain_tool
def search_information(query: str) -> str:
    """提供有关给定主题的事实信息。使用此工具查找答案。"""
    # 实际调用逻辑...
    return result

# 创建智能体
agent = create_tool_calling_agent(llm, tools=[search_information], prompt=...)
executor = AgentExecutor(agent=agent, tools=[search_information])

# 运行 — LLM 自动决定是否需要工具
response = executor.invoke({"input": "法国的首都是什么？"})
```

关键点：
- `@tool` 装饰器把 Python 函数变成 LLM 可调用的工具
- **docstring 极其重要** — LLM 通过描述来决定何时使用这个工具
- `AgentExecutor` 管理"LLM 决策 → 工具执行 → 结果返回"的循环

### CrewAI 方式 — 智能体 + 工具 + 任务

```python
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """获取给定股票代码的最新模拟价格。"""
    return simulated_prices.get(ticker.upper())

agent = Agent(
    role='高级财务分析师',
    tools=[get_stock_price],
)
```

关键点：CrewAI 把工具绑定到有特定"角色"的 Agent 上，更贴近"专业人士使用专业工具"的隐喻。

### Google ADK 方式 — 预构建工具

```python
from google.adk.tools import google_search

agent = Agent(
    name="search_agent",
    model="gemini-2.0-flash",
    tools=[google_search]  # 直接用预构建工具
)
```

ADK 提供的预构建工具：
- `google_search` — Google 搜索
- `BuiltInCodeExecutor` — 沙盒 Python 执行
- `VSearchAgent` — Vertex AI 企业搜索（查询私有数据存储）
- **Vertex Extensions** — 企业级 API 包装器（自动执行，无需手动触发）

---

## "工具调用" vs "函数调用" — 概念区分

书中特别强调了一个概念升级：

| | 函数调用（狭义） | 工具调用（广义） |
|--|---|---|
| **范畴** | 调用一个预定义的代码函数 | 调用任何外部能力 |
| **包括** | Python 函数 | 函数 + API + 数据库 + **其他 Agent** |
| **意义** | 执行一段代码 | 编排整个数字生态系统 |

> 工具可以是另一个智能体 — 这直接连接到第 7 章"多智能体协作"。

---

## 工具描述的重要性

LLM 通过**工具的文字描述**来决定何时使用它。描述写得好不好，直接决定了 Agent 能否正确选择工具：

```python
# 差的描述 — LLM 不知道什么时候该用
@tool
def func(x: str) -> str:
    """处理数据。"""

# 好的描述 — LLM 清楚知道使用条件
@tool
def get_weather(location: str) -> str:
    """获取指定城市的当前天气状况。当用户询问天气、温度或
    是否需要带伞/外套等问题时使用此工具。参数 location 
    应为城市名称，如 'London' 或 '北京'。"""
```

**经验法则**：工具描述要写明 **what（做什么）+ when（什么时候用）+ params（参数格式）**。

---

## 本质理解

工具使用的本质是**让 LLM 充当"调度员"而非"执行者"**：

```
传统编程：程序员写代码 → 计算机执行
工具使用：用户说需求 → LLM 决定调什么 → 框架执行 → LLM 整合结果
```

类比：LLM 像一个**很聪明但没有手的大脑** — 它知道该做什么，但需要"手"（工具）来完成动作。工具使用就是给它接上手。

---

## 与其他章节的关系

```
工具使用 是所有前置模式的"能力放大器"：

提示词链 + 工具使用 = 每步都能调用外部系统
路由 + 工具使用     = 选择用哪个工具（路由的具体应用）
并行化 + 工具使用   = 同时调用多个 API
反思 + 工具使用     = 用代码执行验证输出正确性

工具使用 → 规划（Ch.6）  = 规划的步骤中包含"用什么工具"
工具使用 → MCP（Ch.10）  = MCP 把工具使用标准化为通用协议
工具使用 → 多智能体     = "工具"可以是另一个 Agent
```

---

## 速览（At a Glance）

- **问题背景**：LLM 与外部世界断开，知识静态，无法执行操作或获取实时信息
- **解决方案**：向 LLM 描述可用工具 → LLM 生成结构化调用请求 → 框架执行 → 结果返回给 LLM 整合
- **实践建议**：当 Agent 需要实时数据、私有信息、精确计算、代码执行或触发外部操作时使用
