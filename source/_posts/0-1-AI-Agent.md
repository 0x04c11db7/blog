---
title: 从零开始写 AI Agent：一个 for 循环的演进
date: 2026-4-10 17:32:00
tags: AI
---

### 为什么要从零写一个 Agent？

AI 时代信息爆炸——MCP、RAG、Multi-Agent、Agentic Workflow……新概念一个接一个,每隔几周就有新的框架冒出来。很容易陷进去,感觉 Agent 是一个高深莫测的东西,离自己很远。

但如果你亲手写过一遍(在 AI 帮助下,只需要一个上午),就会发现:**Agent 本身的代码比大佬们写的控制器和转发面简单多了。** 核心逻辑就是一个 for 循环,加上几个工具调用。

这个 Workshop 的目的就是**祛魅**——把地基翻出来看清楚。从一个 223 行的最小 Agent 出发,一步步演进到具备权限控制、任务规划、技能加载、上下文压缩的完整系统。每一版只解决一个问题,每一行代码都有来处。

看完之后,那些概念还会存在,但它们背后的地基你已经摸清楚了。雾里看花,变成近在眼前。
<!-- more --> 

### v1: LLM 调用 + Agent Loop + 单工具 (223行)

最简单的 Agent 实现:一次 LLM 调用 + 一个 while 循环 + 一个计算器工具。核心公式:**Agent = LLM调用 + Loop + 工具执行 + 上下文累积**。

#### 1. LLM 调用接口 (glm.go)

我们用标准 HTTP POST + JSON body 与 LLM 交互,本质上是遵循 **OpenAI Chat Completions API 规范**——这是业界事实标准,GLM、Claude、GPT 等主流模型都兼容它。我们把 `messages`(对话历史)和 `tools`(工具描述)序列化成 JSON 发过去,LLM 服务端收到的只是一段普通的 HTTP 请求体。

有意思的地方在于:**我们从未在 prompt 里告诉 LLM "请用 JSON 格式回复"**,但当我们传入 `tools` 参数时,模型会自动在响应里输出结构化的 `tool_calls` 字段。这是因为模型在训练阶段(SFT + RLHF)就已经被大量"工具调用示例"微调过——它学会了:*一旦上下文里出现工具定义,就应该用约定格式声明调用意图,而不是用自然语言描述*。这个行为是编码在模型权重里的,不依赖任何运行时的格式指令。因此,`json.Unmarshal` 能稳定解析出 `tool_calls`,不是因为我们约束了输出,而是因为模型自己知道该怎么做。这两个 struct 的字段名和 JSON tag 与 API 规范一一对应,Go 的标准库会自动完成映射:

```go
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}
```

```go
func (c *GLMClient) Chat(ctx context.Context, messages []Message, tools []map[string]interface{}) (Message, error) {
	reqBody := map[string]interface{}{
		"model":    "glm-5",
		"messages": messages,
	}
	if len(tools) > 0 {
		reqBody["tools"] = tools
	}

	jsonData, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", c.url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return Message{}, fmt.Errorf("API error: %s", string(body))
	}

	var result struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
	}
	json.Unmarshal(body, &result)

	return result.Choices[0].Message, nil
}
```

#### 2. Agent 主循环 (main.go)

这里有两层循环,职责完全不同。外层是**对话轮次**——用户每发一条消息触发一次;内层才是真正的 **Agent Loop**,处理单条消息内可能发生的多轮工具调用。

LLM 不一定一次就给出最终答案。比如"先算 10+5,再把结果乘以 2",它需要先调用一次 calculator、拿到结果、再调用第二次才能回答。内层 `for i := 0; i < 10` 就是这个循环:有 `tool_calls` 就执行工具、把结果追加进 messages、继续下一轮;没有 `tool_calls` 说明 LLM 已经给出最终答案,直接 return。上限 10 是安全兜底,防止工具持续报错时无限循环——正常路径根本跑不到 10 次。

```go
// 外层:对话轮次循环(main 函数里)
for {
	handleUserMessage(ctx, client, &messages, input)
}

// 内层:单条消息的 Agent Loop
func handleUserMessage(...) {
	for i := 0; i < 10; i++ {  // 上限 10 次,安全兜底
		response, _ := client.Chat(ctx, *messages, tools)
		*messages = append(*messages, response)

		if len(response.ToolCalls) == 0 {
			fmt.Printf("AI: %s\n\n", response.Content)
			return  // 无工具调用 = 最终答案,退出
		}

		// 执行工具,把结果追加进 messages,继续下一轮
		for _, tc := range response.ToolCalls {
			result := calculator(tc.Function.Arguments)
			*messages = append(*messages, Message{Role: "tool", Content: result})
		}
	}
	fmt.Println("超过最大轮次")  // 兜底,正常不会触发
}
```

---

### v2: 多工具系统 (410行)

#### 问题:LLM 只会"说",不会"干"

v1 的 Agent 只有一个计算器——LLM 可以推理、可以规划,但没有办法真正操作文件、执行命令、读取外部数据。它的能力边界就是它的上下文窗口,干不了任何需要副作用的事。

v2 的思路:引入工具系统,让 LLM 通过结构化的 Function Call 驱动真实操作。抽出统一的 `ExecuteTool` 路由层——新增工具只改 tools.go,Agent Loop 完全不动。

#### 工具路由接口 (tools.go)

```go
func ExecuteTool(toolName string, arguments string) string {
	switch toolName {
	case "calculator":
		var args struct {
			Expression string `json:"expression"`
		}
		json.Unmarshal([]byte(arguments), &args)
		return calculator(args.Expression)

	case "read_file":
		var args struct {
			Path string `json:"path"`
		}
		json.Unmarshal([]byte(arguments), &args)
		return readFile(args.Path)

	case "write_file":
		var args struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}
		json.Unmarshal([]byte(arguments), &args)
		return writeFile(args.Path, args.Content)

	case "edit_file":
		var args struct {
			Path    string `json:"path"`
			OldText string `json:"old_text"`
			NewText string `json:"new_text"`
		}
		json.Unmarshal([]byte(arguments), &args)
		return editFile(args.Path, args.OldText, args.NewText)

	case "bash":
		var args struct {
			Command string `json:"command"`
		}
		json.Unmarshal([]byte(arguments), &args)
		return runBash(args.Command)

	default:
		return fmt.Sprintf("未知工具: %s", toolName)
	}
}
```

**关键特性**:统一的 switch 路由,每个工具独立解析 JSON 参数,返回字符串结果。支持 5 种工具:计算器、文件读写、编辑、命令执行。

#### 两个最重要的工具:bash 与 edit_file

有了 `bash`,Agent 拥有了和人类工程师几乎等价的操作能力——编译、运行测试、调用 CLI 工具、查询系统状态,任何能在终端里做的事它都能做。这是从"玩具 Agent"到"能干活的 Agent"的分水岭。

```go
func runBash(command string) string {
	cmd := exec.Command("bash", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("执行失败: %v\n输出: %s", err, string(output))
	}
	return string(output)
}
```

既然有了 bash,为什么还要单独提供 `edit_file`?直接让 Agent 用 `sed` 或 `echo >>` 改文件不行吗?

实践中不行。**LLM 生成的 shell 转义极不可靠**——多行内容、特殊字符、引号嵌套,稍有偏差就静默写错甚至清空文件,而且错误往往难以复现。`edit_file` 把"找到原文、替换为新文"抽象成一个原子操作,参数是结构化的 JSON 字符串而不是需要转义的 shell 命令,LLM 生成起来可靠得多。本质上是给 LLM 设计**适合它使用的接口**,而不是把人类用的工具直接暴露出去。

```go
func editFile(path, oldText, newText string) string {
	content, _ := os.ReadFile(path)
	if !strings.Contains(string(content), oldText) {
		return fmt.Sprintf("错误:在文件中找不到指定的文本")  // 明确报错,不静默失败
	}
	newContent := strings.Replace(string(content), oldText, newText, 1)
	os.WriteFile(path, []byte(newContent), 0644)
	return fmt.Sprintf("成功编辑文件: %s", path)
}
```

注意 `strings.Contains` 的前置检查——找不到原文直接报错,而不是什么都不做返回成功。这对 Agent 很重要:**模糊失败比明确报错更难恢复**,LLM 看到错误信息才能在下一轮修正。

#### 集成到 Agent Loop (main.go)

```go
// v1: 硬编码 calculator
// result := calculator(args.Expression)

// v2: 通用工具路由 ⭐
for _, tc := range response.ToolCalls {
	fmt.Printf("[调用工具: %s]\n", tc.Function.Name)
	fmt.Printf("[参数: %s]\n", tc.Function.Arguments)

	// ⭐ 关键变化:统一调用 ExecuteTool,不再硬编码工具名
	result := ExecuteTool(tc.Function.Name, tc.Function.Arguments)
	fmt.Printf("[工具结果: %s]\n", result)

	*messages = append(*messages, Message{
		Role:    "tool",
		Content: result,
		// ...
	})
}
```

**核心变化**:Agent Loop 本身几乎不变,只是把 `calculator(args.Expression)` 替换为 `ExecuteTool(toolName, arguments)`。这实现了**开闭原则**:新增工具只需修改 tools.go,main.go 无需改动。

---

### v3: 权限系统 (607行)

#### 问题:Agent 拿到了 bash,但没有任何约束

v2 给了 Agent bash 和文件写入能力,但这意味着一条错误的指令就能删文件、跑危险命令。权限系统要解决的核心问题是:**Agent 应该在什么情况下可以自主行动,什么情况下必须先问人**。

v3 定义了 5 种权限模式,覆盖从"完全信任"到"完全只读"的整个光谱:

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `readonly` | 只允许读取和计算 | 代码审查、只读分析 |
| `ask_on_write` | 读自动通过,写操作弹出确认 | 日常使用推荐默认值 |
| `auto_write` | 文件读写自动通过,bash 执行需确认 | 批量文件处理 |
| `allow_all` | 全部放行,无任何拦截 | 完全信任的自动化流水线 |
| `plan_mode` | 禁止一切写操作,只允许读和分析 | 让 Agent 先规划再执行(v4 会用到) |

另外还有一层路径检查:所有文件操作必须在指定 workspace 目录内,`filepath.Rel` 检测到 `..` 路径穿越直接拒绝。这是最基本的沙箱隔离。

**这是简化版本。**生产级权限系统会复杂得多:细化到单个工具粒度的白名单、操作审计日志、会话级权限动态升降级、人工审批工作流……我们这里的 5 模式分层已经能覆盖大多数教学和原型场景,但如果是面向真实用户的 Agent,权限设计本身会是一个独立的工程课题。

#### 权限检查逻辑 (permission.go)

```go
func (p *Permission) Check(opType OperationType, toolName string, details string) (bool, string) {
	switch p.Mode {
	case ModeReadOnly:
		if opType == OpSafe || opType == OpRead {
			return true, ""
		}
		return false, fmt.Sprintf("权限拒绝:当前为只读模式,不允许 %s 操作", opType)

	case ModeAskOnWrite:
		if opType == OpSafe || opType == OpRead {
			return true, ""
		}
		return p.askUser(toolName, details)

	case ModeAutoWrite:
		if opType == OpExecute {
			return p.askUser(toolName, details)
		}
		return true, ""

	case ModeAllowAll:
		return true, ""

	case ModePlan:
		if opType == OpSafe || opType == OpRead {
			return true, ""
		}
		return false, fmt.Sprintf("权限拒绝:计划模式禁止 %s 操作", opType)

	default:
		return false, "未知权限模式"
	}
}
```

**核心机制**:策略模式,5 种模式分别对应不同安全等级。ModeAskOnWrite 需交互确认,ModePlan 完全禁止修改。返回 (bool, string) 元组:布尔值表示是否允许,字符串为拒绝理由。

#### 集成到 Agent Loop

```go
// Agent Loop 代码与 v2 完全相同!权限检查在工具函数内部透明处理

// tools.go - 以 writeFile 为例
func writeFile(path, content string) string {
	// ⭐ 步骤1:检查路径是否在工作区内
	if err := globalPermission.CheckPath(path); err != nil {
		return fmt.Sprintf("权限错误: %v", err)
	}

	// ⭐ 步骤2:根据权限模式检查是否允许写操作
	allowed, reason := globalPermission.Check(
		OpWrite,
		"write_file",
		fmt.Sprintf("写入文件: %s (内容长度: %d 字节)", path, len(content)),
	)
	if !allowed {
		return reason  // 返回拒绝原因,如 "权限拒绝:只读模式"
	}

	// ⭐ 步骤3:通过检查后,执行实际操作
	err := os.WriteFile(path, []byte(content), 0644)
	// ...
}
```

**设计亮点**:权限检查对 Agent Loop **完全透明**。Loop 调用 `ExecuteTool()`,工具内部自行处理权限,被拒绝时返回错误信息而非执行。这是**职责分离**的典范。

---

### v4: Todo + Plan Mode (1119行)

#### 问题:多步骤任务里 LLM 容易迷失

v3 的 Agent 面对复杂任务时,往往走几步就忘了目标,或者反复做同样的事。根本原因是:**LLM 没有跨轮次的持久记忆**,每轮对话只能靠 context 里的历史消息推断"我做到哪了"——历史越长越贵,越短越容易失忆。

v4 的思路:引入 todo 工具把任务状态外化到结构化列表,再通过 System Prompt 和提醒注入告诉 LLM 如何使用它。

#### 1. Plan Mode 的起点:System Prompt 注入 (main.go)

v3 的 `plan_mode` 只是权限层的拦截——写操作会被拒绝,但 LLM 自己不知道它处于"规划模式",只会看到一堆权限错误然后困惑地重试。v4 补上这个缺口:**启动时根据当前模式动态构建 System Prompt**,把规划流程写进 LLM 的行为指令。权限拦截是"墙",Prompt 是"路牌"——告诉 LLM 应该怎么走,而不只是哪里不能去。

```go
// main.go — 根据模式动态构建 system prompt
basePrompt := `你是一个助手,可以使用工具:calculator、read_file、write_file、edit_file、bash、todo。
重要:对于复杂任务,使用 todo 工具进行任务分解和进度跟踪:
1. 收到复杂任务时,先用 todo add 添加子任务
2. 开始任务前,用 todo start 标记
3. 完成任务后,用 todo complete 标记`

var systemPrompt string
if mode == ModePlan {
	systemPrompt = basePrompt + `

特别注意 - Plan Mode(计划模式):
Plan mode 已激活。你必须遵循以下三阶段流程:

重要约束:只能使用 calculator、read_file、todo
禁止使用:write_file、edit_file、bash

Phase 1 - 理解需求:使用 read_file 阅读相关代码,有疑问直接提问,不要猜测
Phase 2 - 任务分解:用 todo add 添加子任务,要求具体明确、原子化、可验证
  ✅ "在 snake.html 中创建 canvas 元素,设置 id='gameCanvas',宽 800px 高 600px"
  ❌ "创建游戏界面"(太笼统)
Phase 3 - 向用户确认:说明任务数和思路,告知使用 /execute 执行,等待确认`
} else {
	systemPrompt = basePrompt
}

messages := []Message{
	{Role: "system", Content: systemPrompt},  // ⭐ 运行时注入
}
```

#### 2. 为什么需要 todo 工具?

Prompt 告诉 LLM"先制定计划",但有一个根本问题:**LLM 没有跨轮次的持久记忆**。每次调用都只能看到 messages 里的内容——如果计划只存在于某条 assistant 消息的文字里,随着对话推进它会被"淹没",LLM 很容易遗忘或跳过还没完成的任务。

解决方案是把"记住任务"这件事变成一个**工具调用行为**,而不是依赖 LLM 的上下文注意力。LLM 主动调用 `todo add` 把任务写入外部状态,调用 `todo start/complete` 更新进度——状态由 Go 程序维护,不会消失,还可以在需要时重新注入回 messages。

#### 3. Todo 管理器 (todo.go)

```go
type TodoManager struct {
	Items               []TodoItem
	NextID              int
	RoundsSinceLastTodo int  // 追踪多少轮未用 todo 工具
}

func (tm *TodoManager) AddItem(description string) TodoItem {
	item := TodoItem{ID: tm.NextID, Description: description, Status: "pending"}
	tm.Items = append(tm.Items, item)
	tm.NextID++
	tm.RoundsSinceLastTodo = 0  // 重置计数
	return item
}

// pending → doing → done 三阶段流转
func (tm *TodoManager) StartItem(id int) error { ... }
func (tm *TodoManager) CompleteItem(id int) error { ... }
```

状态完全在 Go 侧维护,LLM 通过工具调用读写,不依赖上下文记忆。

#### 4. 提醒注入到 Agent Loop (main.go) ⭐ 核心变化

```go
// v4 是唯一真正修改 Agent Loop 的版本!

for _, tc := range response.ToolCalls {
	result := ExecuteTool(tc.Function.Name, tc.Function.Arguments)

	// ⭐ 如果 3 轮未使用 todo 工具,把待办状态注入到工具结果里
	// LLM 下一轮会读到它,从而想起来更新任务进度
	if globalTodoManager.ShouldRemind() && tc.Function.Name != "todo" {
		result += globalTodoManager.GetReminder()
	}

	*messages = append(*messages, Message{Role: "tool", Content: result, ...})
}
globalTodoManager.IncrementRound()
```

注入的内容是什么?就是下面这段拼出来的纯文本,追加在工具结果末尾,LLM 读 tool message 时会自然看到它:

```go
func (tm *TodoManager) ShouldRemind() bool {
	return tm.RoundsSinceLastTodo >= 3 && len(tm.Items) > 0
}

func (tm *TodoManager) GetReminder() string {
	hasPending, hasDoing := false, false
	for _, item := range tm.Items {
		if item.Status == "pending" { hasPending = true }
		if item.Status == "doing"   { hasDoing = true }
	}

	var sb strings.Builder
	sb.WriteString("\n🔔 待办提醒:\n")
	if hasDoing   { sb.WriteString("  有任务正在进行中,记得更新状态\n") }
	if hasPending { sb.WriteString("  有待处理的任务,记得开始执行\n") }
	sb.WriteString("  使用 todo 工具查看完整列表\n")
	return sb.String()
}
```

提醒注入是闭环的最后一环:Prompt 建立初始行为 → todo 工具外化状态 → 提醒注入在 LLM 走神时把状态送回它的视野。三者缺一不可——只有 Prompt 容易遗忘,只有 todo 工具没有提醒 LLM 会忽视,只有提醒没有工具则状态无处存放。

---

### v5: Skill 系统 (1341行)

#### 问题:System Prompt 的扩展瓶颈

随着 Agent 能力扩展,一个自然的冲动是把更多知识塞进 System Prompt——"如何处理 PDF"、"代码审查标准"、"数据库操作规范"。但这条路走不远:每次对话都要发送全部内容,大量 token 花在用不到的知识上;更根本的是,context window 有硬上限,知识越多越快撞墙。

v5 的思路:把专项能力写成独立的 `SKILL.md` 文件,**让 LLM 自己决定什么时候加载什么技能**,而不是由我们预判塞满。

#### SKILL.md 格式

每个技能是一个目录下的 `SKILL.md`,YAML frontmatter 存元数据,分隔符后是完整内容:

```markdown
---
name: code-review
description: Review code quality, style, and best practices
---

# Code Review Skill

### Capabilities
1. Analyze code for bugs and logic errors
2. Review naming conventions and readability
...
```

格式刻意简单:frontmatter 只需 `name` 和 `description` 两个字段,正文是普通 Markdown。

#### 第一步:启动时只扫描元数据

Agent 启动时,`scanSkills()` 遍历 `skills/` 目录,对每个 `SKILL.md` 只读 frontmatter,**不读正文**:

```go
func (sm *SkillManager) scanSkills() {
	entries, _ := os.ReadDir(sm.SkillsDir)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		skillPath := filepath.Join(sm.SkillsDir, entry.Name(), "SKILL.md")
		if _, err := os.Stat(skillPath); err == nil {
			metadata, err := sm.parseSkillMetadata(skillPath)
			if err != nil {
				continue
			}
			sm.AvailableSkills[entry.Name()] = &Skill{Metadata: metadata, Loaded: false}
		}
	}
}

func (sm *SkillManager) parseSkillMetadata(skillPath string) (SkillMetadata, error) {
	content, _ := os.ReadFile(skillPath)
	text := string(content)
	// text[4:] 跳过开头的 "---\n"
	parts := strings.SplitN(text[4:], "\n---\n", 2)
	// parts[0] 是 YAML frontmatter,parts[1] 是正文(不读)
	// 手动解析 name: / description: 两行
	...
}
```

`parseSkillMetadata` 解析到 `\n---\n` 就停下,正文根本不进内存。`AvailableSkills` 里存的只是名字和一句描述。

#### 第二步:把元数据摘要注入 System Prompt

LLM 需要知道"有哪些技能可以用",但不需要知道细节。`GetAvailableSkillsSummary()` 把元数据格式化成几行文字,拼进 system prompt:

```go
// main.go
basePrompt := `你是一个助手,可以使用工具:...

` + globalSkillManager.GetAvailableSkillsSummary() + `
...`

// GetAvailableSkillsSummary() 输出示例:
// 可用技能:
//   - pdf: Process and extract information from PDF files [未加载]
//   - code-review: Review code quality, style, and best practices [未加载]
// 使用 load_skill 工具加载完整技能内容
```

每次对话多出的 token 消耗只有这几行摘要,而不是所有技能的完整文档。

#### 第三步:LLM 按需调用 load_skill

当 LLM 判断需要某个技能时,调用 `load_skill` 工具。`LoadSkill()` 这时才读完整文件,分割出正文,返回内容:

```go
func (sm *SkillManager) LoadSkill(skillName string) (string, error) {
	skill, exists := sm.AvailableSkills[skillName]
	if !exists {
		return "", fmt.Errorf("技能不存在: %s", skillName)
	}
	if skill.Loaded {
		return fmt.Sprintf("技能 '%s' 已经加载", skillName), nil  // 幂等
	}

	content, _ := os.ReadFile(filepath.Join(sm.SkillsDir, skillName, "SKILL.md"))
	text := string(content)
	parts := strings.SplitN(text[4:], "\n---\n", 2)

	skill.Content = parts[1]  // 只取正文,去掉 frontmatter
	skill.Loaded = true
	sm.LoadedSkills[skillName] = skill

	return fmt.Sprintf("✅ 技能 '%s' 已加载\n\n%s", skillName, skill.Content), nil
}
```

返回值直接作为工具结果追加进 `messages`,技能内容就此进入对话上下文——LLM 下一轮就能用上。注意幂等检查:重复加载直接返回,不会重复注入。

#### Agent Loop 完全透明

v5 对 Agent Loop 本身没有任何改动。`load_skill` 只是 `ExecuteTool` 里新增的一个 case:

```go
// tools.go - ExecuteTool 新增一个分支,仅此而已
if toolName == "load_skill" {
	return loadSkillTool(arguments)
}
```

Loop 不感知"技能"的存在,只是照常把工具返回值追加进 messages。技能内容经由这条普通路径自然流入上下文,不需要任何特殊协议。新增技能也不需要改任何代码——放一个目录进去,重启后自动被扫描到。

---

### v6: 上下文压缩 (1590行)

#### 问题:对话越长,成本越高,直到崩溃

LLM 的 API 调用按 token 计费,而且每次调用都要发送**完整的历史消息**。随着对话轮次增加,token 消耗线性增长——一个长任务里 bash 命令的输出、文件内容的读取结果会迅速撑大上下文。更根本的问题是 context window 有硬上限,超出直接报错,对话无法继续。

v6 引入 `ContextManager` 统一管理所有消息的读写。Agent Loop 里所有 `*messages` 的直接操作,全部改成走 `globalContextManager.AddMessage()` 和 `globalContextManager.GetMessages()`,压缩逻辑在这个统一出入口里自然嵌入。

#### 策略一:CompressMicro(每轮自动、零成本)

`GetMessages()` 是 messages 流向 LLM 的唯一出口,每次调用都先触发 `CompressMicro`。它扫描所有 role 为 `tool` 的消息,只保留最近 `KeepRecentRounds` 条,把更早的替换成占位符,零 API 调用静默执行:

```go
func (cm *ContextManager) GetMessages() []Message {
	cm.CompressMicro()
	return cm.Messages
}

func (cm *ContextManager) CompressMicro() {
	toolIndices := []int{}
	for i := len(cm.Messages) - 1; i >= 0; i-- {
		if cm.Messages[i].Role == "tool" {
			toolIndices = append([]int{i}, toolIndices...)
		}
	}
	if len(toolIndices) <= cm.KeepRecentRounds {
		return
	}
	for _, idx := range toolIndices[:len(toolIndices)-cm.KeepRecentRounds] {
		msg := &cm.Messages[idx]
		if !strings.HasPrefix(msg.Content, "[已压缩的工具结果]") {
			msg.Content = "[已压缩的工具结果]"
		}
	}
}
```

代价是彻底丢失历史工具输出的细节——对 LLM 而言,那些工具调用"发生过"但结果已不可见。Micro 只截断工具结果,整体消息数量不减少,对话仍会持续增长。

#### 策略二:CompressAuto(超限触发、保留语义)

当 `NeedsCompression()` 检测到 token 估算超过阈值(`MaxTokens × CompressionRatio`,默认 8192 × 0.9),Agent Loop 在每次 Chat 前自动触发;用户也可随时 `/compact` 手动触发。核心是把待压缩的历史段用一次独立的 LLM 调用浓缩成摘要,再替换原始消息:

```go
func (cm *ContextManager) CompressAuto(ctx context.Context, client *GLMClient) error {
	compressEndIdx := len(cm.Messages) - cm.KeepRecentRounds*2
	toSummarize := cm.Messages[systemMsgIdx+1 : compressEndIdx]

	response, _ := client.Chat(ctx, []Message{
		{Role: "user", Content: buildSummaryPrompt(toSummarize)},
	}, nil)

	cm.Messages = []Message{
		cm.Messages[systemMsgIdx],                                       // 保留 system prompt
		{Role: "assistant", Content: "📋 **对话摘要**\n\n" + response.Content}, // 摘要替换历史
		cm.Messages[compressEndIdx:]...,                                 // 保留最近 N 轮
	}
	return nil
}
```

摘要的质量取决于提示词。`buildSummaryPrompt` 把历史消息序列化后,要求 LLM 输出结构化的两段:

```go
sb.WriteString("输出格式:\n")
sb.WriteString("<analysis>\n对当前状态的深入分析(2-3 句)\n</analysis>\n")
sb.WriteString("<summary>\n")
sb.WriteString("1. 初始用户请求和高层目标\n")
sb.WriteString("2. 已完成的关键步骤\n")
sb.WriteString("3. 当前工作状态\n")
sb.WriteString("4. 待完成的任务(如有)\n")
sb.WriteString("5. 重要的技术决策和原因\n")
sb.WriteString("6. 相关文件路径\n")
sb.WriteString("7. 遇到的错误和解决方案(如有)\n")
sb.WriteString("8. 下一步行动建议\n")
sb.WriteString("</summary>\n")
```

结构化输出让 LLM 在后续轮次能快速定位上下文,而不是从一段流水叙述里自己提取。这也是为什么摘要以 `assistant` 角色注入——对 LLM 而言,这就是它"自己之前写下的记录",而不是外部注入的元数据。

---

### v7~vN:还能走多远?

我们用 ~1600 行 Go 代码,从零实现了一个具备工具调用、权限控制、任务规划、技能加载和上下文压缩的 Agent。但这只是起点。现实中的 Agent 系统还有很多值得继续演进的方向:

#### 待完善的能力

| 方向 | 描述 |
|------|------|
| **子 Agent** | 主 Agent 把子任务委托给独立 Agent 并发执行,收摘要后继续 |
| **多 Session** | 同时维护多个独立对话上下文,支持并行任务或多用户场景 |
| **长期 Memory** | 跨 Session 持久化关键知识,下次对话不从零开始 |
| **流式输出** | Streaming API 逐 token 返回,用户体验从"等待"变成"看着它思考" |

#### 但核心思想其实很简单

回头看整个演进,Agent 的本质始终是同一个 for 循环:**调用 LLM → 执行工具 → 把结果塞回上下文 → 重复**。所有复杂性都是在这个循环的外围加约束、加状态、加策略。

以子 Agent 为例——听起来很复杂,实现思路其实直接:把"启动子 Agent"做成一个普通工具,主 Agent 通过 Function Call 调用它;工具实现里用 Go 的 goroutine 跑一个完整的 Agent Loop,完成后把执行摘要作为 tool result 返回;主 Agent 拿到摘要继续规划下一步。整个机制复用了我们已有的工具系统、上下文压缩和 todo 状态管理,没有引入任何新的架构概念。

多 Session、Memory 也是同理:Session 是上下文的隔离单元,Memory 是跨 Session 的持久化 Skill,它们都能用已有的抽象自然延伸出来。

#### 真正的难点:Harness

把 Agent Loop 写出来只需要几十行。让 Agent 真正**把工作做好**,才是难的部分——这就是 **Harness** 的概念:围绕 Agent 构建的一整套"脚手架",决定了 LLM 能看到什么、能做什么、做错了怎么纠正。

我们的实现其实已经包含了 Harness 的雏形:

| 我们实现的 | 对应的 Harness 思想 |
|-----------|---------------------|
| 动态 System Prompt + Plan Mode | 行为约束:告诉 LLM 在当前上下文里应该怎么行动 |
| Todo 工具 + GetReminder 注入 | 状态跟踪:把任务进度持续喂给 LLM,防止它"忘事" |
| Skill 按需加载 | 知识路由:只在需要时注入相关上下文,避免噪音 |
| 权限系统 + ask_on_write | 人机协作:在关键决策点把控制权交还给人 |
| ContextManager + 压缩策略 | 上下文管理:控制 LLM 每次"看到"的信息密度 |

但生产级的 Harness 远不止这些——评估与回滚(LLM 做错了怎么恢复)、可观测性(每一步的 token 消耗、工具调用链路)、沙箱隔离(bash 执行的安全边界)、重试与降级……每一项都是独立的工程课题。

从这个角度看,Agent 本身的代码是最简单的部分。**Harness 才是让 Agent 真正可用的工程核心。**

这也是为什么理解 Agent 的最好方式,是自己从零写一遍——当你亲手把 for 循环、工具路由、权限检查、提示词注入这些拼在一起,"Agent"就从一个模糊的概念变成了一组具体的设计决策。
