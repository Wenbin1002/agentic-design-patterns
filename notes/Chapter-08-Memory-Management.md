# 第 8 章：记忆管理（Memory Management）

> **一句话理解**：给 Agent 加上"记忆" — 短期记忆保持当前对话连贯，长期记忆让 Agent 跨会话保留信息、学习和个性化。没有记忆的 Agent 每次对话都像第一次见面。

---

## 为什么需要它？

前 7 章的 Agent 有一个根本缺陷：**无状态** — 每次交互都是独立的，不记得之前发生了什么。

```
用户第一次: "我叫张三，我对 AI 很感兴趣"
Agent: "你好张三！AI 是很有趣的领域..."

用户第二次: "上次我们聊了什么？"
Agent: "抱歉，我不知道我们之前聊过什么。" ← 没有记忆
```

---

## 核心概念：两种记忆

| | 短期记忆（上下文记忆） | 长期记忆（持久记忆） |
|--|---------|---------|
| **类比** | 工作记忆 / 便签纸 | 笔记本 / 知识库 |
| **存储位置** | LLM 上下文窗口内 | 外部数据库（向量库/知识图谱） |
| **生命周期** | 当前会话，会话结束即丢失 | 跨会话持久存在 |
| **容量** | 受上下文窗口限制（有限） | 理论上无限 |
| **访问方式** | 直接在提示词中 | 语义搜索检索后注入提示词 |
| **内容** | 最近消息、工具结果、中间推理 | 用户偏好、历史事实、学习经验 |

---

## Google ADK 的三层架构

ADK 把记忆管理结构化为三个层次：

```
Session（会话）── 一个独立的聊天线程
  └── Events ── 消息的时间顺序记录
  └── State  ── 该会话的临时数据字典
      ├── user:xxx  → 关联到用户，跨会话持久
      ├── app:xxx   → 应用全局共享
      └── temp:xxx  → 仅当前轮次有效

Memory（记忆）── 可搜索的长期知识库
  └── 从历史会话中提取的信息
  └── 通过语义搜索检索
```

**关键原则**：通过 `append_event` 的 `state_delta` 更新状态，而不是直接修改 `session.state` — 确保可追溯和正确持久化。

---

## LangChain/LangGraph 的记忆方案

### 短期记忆

```python
# 最基础：ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

chain.predict(question="我叫 Sam")  # 记住了
chain.predict(question="我叫什么？")  # → "你叫 Sam"
```

### 长期记忆 — 三种类型

| 类型 | 含义 | 实现方式 | 示例 |
|------|------|---------|------|
| **语义记忆** | 记住事实 | 用户 profile / JSON 文档 | "用户偏好简短回答" |
| **情景记忆** | 记住经历 | 少样本示例 | "上次用户问 X 时，成功做法是 Y" |
| **程序记忆** | 记住规则 | 系统提示词（可自我更新） | Agent 通过反思修改自身指令 |

```python
# LangGraph 长期记忆存储
store = InMemoryStore(index={"embed": embed_fn, "dims": 768})
store.put(namespace, "user-prefs", {"rules": ["用户喜欢简短回答"]})

# 检索
items = store.search(namespace, query="用户偏好")
```

---

## 程序记忆 + 反思 = Agent 自我进化

这是最有意思的组合 — Agent 可以**修改自己的系统提示词**：

```python
def update_instructions(state, store):
    current_instructions = store.get(("agent_instructions",), "agent_a")
    
    # 让 LLM 反思对话并改进指令
    new_instructions = llm.invoke(f"""
        当前指令：{current_instructions}
        最近对话：{state["messages"]}
        请改进指令使 Agent 表现更好。
    """)
    
    store.put(("agent_instructions",), "agent_a", new_instructions)
```

这直接连接到第 9 章"学习与适应"。

---

## 本质理解

记忆管理的本质是解决 LLM 的**无状态问题**：

```
LLM 本身：无状态函数 f(input) → output，不记得之前的调用
+ 短期记忆：把历史消息塞进 input → f(history + new_input) → output
+ 长期记忆：从外部数据库检索相关信息塞进 input → f(retrieved + new_input) → output
```

所有"记忆"本质上都是**想办法把相关信息塞进有限的上下文窗口**。区别只在于"从哪里取"和"取什么"。

---

## 与其他章节的关系

```
记忆管理 是多个模式的基础设施：

记忆 + 反思（Ch.4）  = 基于历史做更好的自我评估
记忆 + 规划（Ch.6）  = 基于过往经验优化未来计划
记忆 → 学习（Ch.9）  = 记忆是学习的前提，学习是记忆的高级应用
记忆 → RAG（Ch.14）  = 长期记忆的典型实现就是 RAG
记忆 + 多智能体     = Agent 间如何共享状态和知识
```

---

## 速览（At a Glance）

- **问题背景**：缺乏记忆的 Agent 无法维护对话上下文、学习或个性化，被限制在一次性交互中
- **解决方案**：双组件系统 — 短期记忆（上下文窗口内）维护对话流 + 长期记忆（外部向量库）支持语义检索
- **实践建议**：当 Agent 需要多轮对话连贯性、多步骤任务跟踪、或基于历史个性化时使用
