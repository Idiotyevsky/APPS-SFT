# Coding Agent 数据合成规格

> 文件：`DATA_SYNTHESIS_SPEC.md`  
> 状态：完整实现规范  
> 目标读者：负责实现数据合成、轨迹验证与 SFT 导出的工程人员  
> 默认模型：`Qwen/Qwen2.5-Coder-7B-Instruct`；开发期可用 3B 同系列模型做 smoke test

## 1. 目标与非目标

本项目基于 APPS 题目构造 Coding Agent 的监督微调数据。数据来源同时覆盖：人工可控的单点错误、自动组合的多点错误，以及模型自然生成的错误。训练目标是让模型学习以下决策：

1. 已知候选代码失败时，何时可以直接修复并提交；
2. 何时需要重放 grader 返回的失败输入以获得程序实际行为；
3. 何时应在首次提交前主动设计测试输入并运行候选程序；
4. 候选解已经可靠时，何时应直接提交而不产生无效工具调用。

完整数据集只使用两个工具，工具名、参数名和语义严格固定：

```text
run_candidate(input)
submit(code)
```

不实现文件系统、Shell、任意 Python scratchpad、`inspect_state`、多样例批量执行或第三种工具。`run_candidate` 始终运行环境中的“当前候选程序”；`submit` 接收并评测一份完整代码。

数据合成的核心对象不是一段看起来像 Agent 的对话，而是：

```text
可观察状态 -> 在该状态下有验证依据的最佳动作
```

所有分类必须由真实执行和反事实分支决定，不能依据 mutation 类型直接指定，也不能为了满足预设比例而修改标签。

### 1.1 一眼看懂完整流水线

```text
APPS train
  -> 清洗题目与测试
  -> 验证 reference solution
  -> 按题目划分 SFT / RL / behavior-dev
  -> 生成候选程序
       ├─ 单点 AST 错误
       ├─ 2～4 点 AST 组合错误
       └─ 模型自然生成的正确解与错误解
  -> 用私有 grader 判断候选真实状态
  -> 对失败候选运行“有执行信息 / 无执行信息”配对实验
  -> 保留有明确证据的动作轨迹
  -> 切分多轮状态，屏蔽所有错误代码对应的 loss
  -> 重放、去重、泄漏检查
  -> 导出 SFT JSONL、metadata 和 QA 报告
```

最终数据不是单纯的“修 bug 代码对”，而是四种可读行为的集合：直接修复、重放失败样例、提交前主动测试、已有可靠解时直接提交。每种行为都可以来自不同难度的题目；前三种还可以覆盖单点错误、多点错误和模型自然错误。

## 2. 不可违反的原则

1. 错误代码只能作为上下文，绝不能作为 SFT target。
2. 只有希望模型在推理时复现的 assistant 动作才计算 loss。
3. `submit` 只返回通过率和至多一个失败测试输入；不得返回 expected output、actual output、reference solution 或 mutation 信息。
4. `run_candidate` 可返回当前候选程序在指定输入上的 stdout、stderr 和错误信息；不得访问 grader 的隐藏测试或标准输出。
5. “提交失败后重放失败样例”的 `run_candidate` 输入必须等于最近一次 `submit` 返回的 failing case。
6. “首次提交前主动验证”的测试输入必须由模型仅根据题面和当前 candidate 自主提出，不能由 hidden tests、expected outputs、reference solution 或 mutation label 派生后注入。
7. 行为类型必须先按反事实效用独立判定，再按本文的目标数量进行采样。只能扩大候选池或对已通过样本下采样，不能为了凑比例修改标签或放宽阈值。
8. reasoning（若保留）只能引用模型可观察的信息。默认不导出长链式推理，只导出工具动作和完整代码。
9. 单点合成错误、多点合成错误和模型自然错误必须全部进入候选生成流程；三者使用同一套真实 grader 和反事实判定，不能因来源不同降低验收标准。
10. APPS test split 不参与合成、调参或门测；SFT、RL 和 behavior-dev 在 APPS train 内按 id 互斥。
11. 所有最终保留轨迹必须以 `submit(code)` 得到 100% 通过结束。

## 3. APPS 数据源、字段与清洗

### 3.1 输入字段

使用 [`codeparrot/apps`](https://huggingface.co/datasets/codeparrot/apps) 的 train split。官方数据卡显示 train/test 各 5000 题、合计 10000 题和 131777 个测试样例；train 中有 195 条没有测试，因此“存在记录”不等于“可以进入合成”。正式实现必须读取原始 train 文件或固定 revision 的等价 Parquet 转换，不能从网页展示内容构建数据集。

官方文件：

- [`train.jsonl`](https://huggingface.co/datasets/codeparrot/apps/blob/main/train.jsonl)：约 107 MB，5000 行；
- [`test.jsonl`](https://huggingface.co/datasets/codeparrot/apps/blob/main/test.jsonl)：约 1.29 GB，5000 行，仅用于最终封存评测；
- 官方 dataset viewer 因仓库包含 loading script 而禁用；开发时可以使用字段相同的 Parquet 镜像查看样例，但正式输入应记录原仓 revision 和 SHA256。

每条原始记录预期包含：

| 字段 | 类型 | 用途 |
|---|---|---|
| `id` | integer/string | 稳定任务 ID、划分与去重键 |
| `question` | string | 提供给模型的题面 |
| `solutions` | JSON string | 解析后为一个或多个 Python reference solutions |
| `input_output` | JSON string | 解析后包含 `inputs`、`outputs`，部分题含 `fn_name` |
| `difficulty` | string | `introductory` / `interview` / `competition`，用于分层划分与统计 |
| `starter_code` | string/null | 函数签名、类定义或起始代码上下文 |
| `url` | string/null | 仅元数据，不进入默认训练文本 |

`solutions` 和 `input_output` 必须分别执行一次严格 `json.loads`。解析失败不得静默修补，应记录 reject reason。

### 3.2 I/O 模式规范化

清洗后每题必须归一为以下两种模式之一：

- `stdin`：无 `fn_name`。测试输入转换为程序标准输入字符串；输出按 grader 的 APPS 兼容规则比较。
- `call`：存在非空 `fn_name`。通过受控 harness 调用目标函数；测试输入规范化为 JSON 可序列化的参数列表。

工具接口仍只有 `input` 一个参数。其值统一为字符串：

- `stdin` 模式：原样 stdin 文本；
- `call` 模式：参数列表的 canonical JSON 字符串，例如 `"[[1,2,3], 2]"`。

轨迹上下文必须明确当前题目的输入格式，避免模型猜测字符串如何解释。grader 和 runner 必须复用同一套输入适配器、输出归一化和超时规则。

### 3.3 清洗顺序

对每个原始题目按以下顺序处理：

1. 校验 `id` 唯一且 `question` 非空；
2. 解析 `solutions`，要求得到非空字符串列表；
3. 解析 `input_output`，要求 `inputs`、`outputs` 均为非空列表且长度一致；
4. 识别 `stdin` 或 `call` 模式，并验证每个测试输入可被统一 adapter 接受；
5. 校验 expected outputs 可被统一 comparator 接受；
6. 对 source 做语法检查，剔除无法编译的 reference；
7. 在隔离 grader 中执行 reference，至少选出一份在全部可用测试上 100% 通过的 verified reference；
8. 对同题多个 verified references，默认选择运行稳定且 AST 节点数最少者；保留全部验证结果到内部元数据；
9. 对同一 reference 重放一次以检查确定性；两次结果不一致则整题进入 quarantine；
10. 输出 cleaned record 或带明确 `reject_reason` 的 rejected record。

以下任一情况必须拒绝或隔离：无测试、输入输出数量不匹配、所有 reference 均失败、持续超时、非确定性、依赖网络/宿主文件、输出无法稳定比较、harness 无法加载或题面/测试明显损坏。

### 3.4 数据划分

仅在 APPS train 的 id 上做带固定 seed 的 difficulty-stratified split，默认目标为：

```text
SFT synthesis pool : 1000 problems
RL training pool   : 3500 problems
Behavior dev pool  :  500 problems
```

先按原始 id 划分，再分别清洗；不得因清洗失败将同一题补入另一个 pool。若有效题数不足，只报告实际数量，不破坏互斥性。必须断言：

```text
SFT_IDs ∩ RL_IDs = ∅
SFT_IDs ∩ DEV_IDs = ∅
RL_IDs  ∩ DEV_IDs = ∅
```

APPS test split 始终保持封存。

## 4. 执行环境与工具协议

### 4.1 当前候选程序

每个 episode 维护 `current_candidate`：

- “提交失败后直接修复”和“重放失败样例后修复”：初始化为已被 grader 证实失败的候选程序。候选程序可以来自单点 AST 变异、多点 AST 变异或模型自然生成；历史失败提交在 SFT 中只作为不可训练状态呈现。
- “提交前主动验证”：初始化为尚未向模型公开 grader 反馈的候选程序。合成器可以在后台知道它是否正确，但该信息不得进入模型上下文。
- “已有可靠解时直接提交”：初始无需保存错误 candidate；assistant 可以提交 verified reference solution，也可以提交经 grader 验证的模型生成解。
- 调用 `submit(code)` 时，环境先将 `current_candidate` 更新为该完整代码，再执行 grader。
- 调用 `run_candidate(input)` 时，只能运行现有 `current_candidate`，不能在参数中传代码。

### 4.2 `run_candidate(input)`

固定 JSON Schema：

```json
{
  "type": "function",
  "function": {
    "name": "run_candidate",
    "description": "Run the current candidate program on one model-provided input and return its observable execution result.",
    "parameters": {
      "type": "object",
      "properties": {
        "input": {"type": "string"}
      },
      "required": ["input"],
      "additionalProperties": false
    }
  }
}
```

结构化返回：

```json
{
  "status": "ok | runtime_error | timeout | invalid_input | output_limit",
  "stdout": "...",
  "stderr": "...",
  "error": null,
  "exit_code": 0,
  "truncated": false
}
```

错误时 `error` 为 `{"type": "...", "message": "..."}`。返回值可以包含 candidate 的 actual stdout，因为这是模型主动执行得到的信息；不能附带 expected output 或正确性判定。stdout/stderr 应分别设置字节上限，超限时截断并令 `truncated=true`。

### 4.3 `submit(code)`

固定 JSON Schema：

```json
{
  "type": "function",
  "function": {
    "name": "submit",
    "description": "Submit a complete solution to the private grader.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string"}
      },
      "required": ["code"],
      "additionalProperties": false
    }
  }
}
```

结构化返回：

```json
{
  "status": "accepted | wrong_answer | runtime_error | timeout | compile_error",
  "passed": 7,
  "total": 10,
  "pass_rate": 0.7,
  "failing_input": "..."
}
```

规则：

- `pass_rate = passed / total`，范围 `[0, 1]`；
- 100% 通过时 `failing_input = null`；
- 未通过时，按稳定测试顺序返回第一个失败输入；
- 对 stdin/call 模式使用第 3.2 节的统一字符串表示；
- 不返回 failing case 的 expected、actual、diff、reference、测试索引或 mutation 类型；
- compile/runtime/timeout 情况仍可返回触发失败的输入；若错误发生在测试开始前，`failing_input=null` 并记录状态。

### 4.4 隔离和资源限制

`run_candidate` 与 `submit` 使用相同的代码沙箱基座，但权限必须分离：

- 无网络；
- 无数据集目录、reference 文件或 grader 文件读取权限；
- 独立临时目录，episode 结束后清理；
- 固定 Python 版本和依赖白名单；
- 每 case CPU/墙钟超时、进程数、内存和输出上限；
- 禁止子进程逃逸及跨 episode 状态；
- grader 的 tests/expected outputs 仅在 grader 进程可见；
- 保存内部完整执行日志，但进入模型上下文前必须经过反馈投影器。

## 5. 需要合成的四种行为

本节使用面向读者的完整名称。实现中如需枚举值，统一使用：

```text
post_submit_direct_repair
post_submit_failure_replay
pre_submit_active_validation
direct_submission
```

展示文档、prompt 和训练文本统一使用以上完整名称，不再另设字母代号。

### 5.1 如何做反事实比较

“反事实”不是指把代码里的一个符号换掉。它是指：对同一个题目、同一个候选程序和同一组生成 seed，只改变模型是否获得一次执行观察，然后比较最终修复成功率。

对某一状态 `s`，用 `k` 次独立生成估计：

```text
p_without_run = 不调用 run_candidate 时，最终 submit 100% 的次数 / k
p_with_run    = 得到 run_candidate 结果后，最终 submit 100% 的次数 / k
utility       = p_with_run - p_without_run
```

除随机 seed 外，模型版本、system prompt、可见上下文、temperature、top-p、max tokens 和动作预算必须一致。默认 `k=3`，阈值为：

```text
HIGH = 2/3
LOW  = 1/3
MIN_UTILITY = 1/3
```

大规模运行可以先用一次 greedy 生成预筛，再对可能被接受的状态做三次配对复核。最终标签必须以完整复核结果为准，不能用单次成败替代概率判断。

### 5.2 提交失败后直接修复

这是“grader 已经提供了足够信息，不需要再次运行程序”的数据。

模型可见状态：

```text
题目、starter code、输入输出约定
当前失败候选程序
最近一次 submit 返回的 pass rate 和 failing input
```

保留轨迹：

```text
失败状态
-> submit(完整修复代码)
-> accepted
```

接受条件：

```text
当前 candidate 已被真实 grader 证实失败
AND p_without_run >= HIGH
AND 至少有一条直接修复轨迹可重放并 100% 通过
```

用于测概率的错误修复、错误代码和失败提交只保存在内部审计数据中，不作为 SFT target。

### 5.3 提交失败后重放失败样例

这是“grader 只指出哪里失败，模型需要主动获取自己的程序在该输入上实际做了什么”的数据。

模型可见起点与上一类相同，但目标的第一次工具调用必须是：

```text
run_candidate(input=<最近一次 submit 原样返回的 failing_input>)
```

典型轨迹：

```text
失败状态
-> run_candidate(failing_input)
-> stdout/stderr/error
-> submit(完整修复代码)
-> accepted
```

接受条件：

```text
p_without_run <= LOW
AND p_with_failure_replay >= HIGH
AND p_with_failure_replay - p_without_run >= MIN_UTILITY
AND run_candidate.input 与公开 failing_input 的规范化字节完全一致
AND 至少有一条成功轨迹可重放
```

如果 failing input 无法解析、观察结果被截断到失去意义，或者执行后仍不能稳定修复，则丢弃，不得因为调用过工具就保留。

### 5.4 首次提交前主动设计测试

这是“模型在得到 grader 反馈前，主动检查当前候选程序”的数据。

模型可见状态：

```text
题目、starter code、输入输出约定
当前候选程序
没有任何 submit/grader 反馈
```

测试输入必须由模型仅依据上述信息提出。生成器需要保存 proposal prompt、模型原始输出和解析后的 input，然后真实执行：

```text
run_candidate(model_proposed_input)
```

典型轨迹：

```text
提交前状态
-> run_candidate(模型自主提出的输入)
-> stdout/stderr/error
-> submit(修复后或确认不变的完整代码)
-> accepted
```

对照分支不提供执行结果，要求同一模型从同一状态直接提交或修复后提交。接受条件：

```text
p_without_run <= LOW
AND p_with_active_validation >= HIGH
AND p_with_active_validation - p_without_run >= MIN_UTILITY
AND input 的 provenance 为 model_generated_from_observable_context
AND input 可解析、可执行，且不是合成器从 hidden tests 中挑出后交给模型的
AND 至少有一条成功轨迹可重放
```

proposal prompt builder 只能接收题面、starter code、I/O 说明和 candidate 四类白名单字段。它的函数签名中不应出现 grader result、reference solution、mutation record 或 hidden tests，避免调用方误传。

如果最终代码未修改，只有当 candidate 本身真实 100% 通过时才可保留。仅仅调用了 `run_candidate` 不足以证明这是有价值的主动验证，必须满足反事实增益。

### 5.5 已有可靠解时直接提交

这是“无需额外调试和工具消耗，直接提交可靠完整解”的数据。可靠解可以有两个来源：

1. 已在同一 grader 配置下至少两次 100% 通过的 APPS reference；
2. 模型根据题面生成，并至少两次 100% 通过的自然正确解。

轨迹：

```text
题目
-> submit(可靠完整代码)
-> accepted
```

接受条件：

```text
代码两次 grader 验证均为 accepted
AND 目标轨迹没有 run_candidate
AND 保存后的轨迹再次重放仍为 accepted
```

reference solution 和模型生成正确解都应纳入数据，用 `solution_origin` 区分。不得为了制造主动验证样本，给本来可靠的解插入无效 `run_candidate`。

### 5.6 多轮、多错误轨迹如何处理

多点错误可能无法在一次修改中全部修完。原始 episode 可以包含：

```text
失败状态
-> run_candidate(failing_input)
-> 修复并 submit
-> 仍失败，得到新的 failing input
-> run_candidate(new_failing_input)
-> 再次修复并 submit
-> accepted
```

其中任何未通过的 `submit(code)` 都不能作为训练 target。默认导出仍保持一条 episode 对应一行 JSONL，但对每个 assistant 回合单独设置 loss mask：有效的执行动作和最终正确提交可训练，任何中间失败提交不可训练。

QA 还应把完整 episode 临时切成若干“状态—正确下一动作”视图，用于逐步检查：

```text
样本 1：初始失败状态 -> run_candidate(...)
样本 2：包含第一次修复失败及新反馈的状态 -> run_candidate(...)
样本 3：包含最新观察的状态 -> submit(最终正确代码)
```

这些切片只用于审计，默认不额外计入训练数据条数。这样最终正式数据保持 1200 行，同时仍能验证每个监督动作是否有依据。

同一 episode 可以依次出现“失败样例重放”和“直接修复”行为，因此 metadata 应保存 `behavior_sequence`，不要强迫整条多轮轨迹只有一个字母类别。

### 5.7 模糊与丢弃规则

以下状态不导出到 SFT：

- `p_without_run` 与 `p_with_run` 同时高，无法证明执行是必要动作；
- 两者同时低，执行信息没有带来稳定修复；
- `utility < MIN_UTILITY`；
- 同一状态在重复执行中 failure signature 不稳定；
- 两个反事实分支使用了不同的非 seed 生成参数；
- 轨迹包含非法工具名、额外参数或不完整代码；
- 任何 oracle 信息进入模型可见上下文；
- 最终提交不是 100% 通过；
- 多错误轨迹中的错误提交被错误地标为 trainable。

若一个状态看似满足多种行为，应保留完整分支统计，并选择证据最强、效用 margin 最大的行为。不要复制为多个高度相关样本。

## 6. 候选程序的三种来源

完整数据必须同时包含以下来源：

| 来源 | 作用 | `candidate_origin` |
|---|---|---|
| 单点 AST 错误 | 提供边界清楚、可控、易验证的局部错误 | `synthetic_single` |
| 多点 AST 错误 | 提供需要连续诊断和多轮修复的组合错误 | `synthetic_multi` |
| 模型自然错误 | 覆盖算法、状态设计、输入解析、复杂度等真实失败 | `model_natural_failure` |
| 模型自然正确解 | 扩充“可靠解直接提交”，减少 reference 风格偏移 | `model_natural_correct` |

三种错误来源进入同一个候选队列，使用完全相同的 grader、工具协议、反事实分类和 SFT 导出规则。最终数据按第 6.7 节的来源配额采样，但配额只能作用于已经通过全部验证的候选，不能影响行为判定或质量门槛。

### 6.1 单点 AST 错误

输入只能是 verified reference。每个单点 mutant 必须满足：

- Python 可解析、可编译；
- 与 reference 不同；
- 从 reference AST 到 mutant AST 恰好一个受支持的语义编辑；
- 除格式化外无第二处语义变化；
- grader 通过率默认满足 `0 < pass_rate < 1`；
- 重放得到相同 pass rate 和 failing input；
- 不是只改变注释、变量名、空白或死代码。

支持的 mutation operators 至少包括：

| family | 示例 | 约束 |
|---|---|---|
| comparator | `<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=` | 只替换一个 `Compare` op |
| arithmetic | `+` ↔ `-`, `*` ↔ `//`, `%` 删除或增加 | 保证类型仍合理 |
| boolean | `and` ↔ `or`, 单个条件取反 | 不同时改多个条件 |
| boundary constant | `0` ↔ `1`, 常量偏移 `±1` | 只改一个 literal/offset |
| range bound | `range(n)` ↔ `range(n-1/n+1)` | 只改一个 start/stop |
| index offset | `a[i]` ↔ `a[i-1/i+1]` | 过滤普遍越界样本 |
| initialization | DP、计数器、答案初值错误 | 只改一个赋值 RHS |
| update target | `dp[i]` 写到相邻位置 | 静态确认作用域和类型 |
| aggregation | `min` ↔ `max`, `sum` ↔ `len` | 仅限可验证的 builtin 调用 |
| loop direction | 正向与逆向更新错误 | 单处修改 iterator 参数 |
| return expression | 返回邻近变量或偏移值 | 不删除整个 return |
| input/output handling | 少读一项、错误 join/separator | 必须保持 harness 可执行 |

mutation family 只用于生成和分析，绝不能直接决定应该调用什么工具。

### 6.2 自动组合多点错误

多点错误由同一 reference 上的多个兼容 AST edit 自动组合，不靠人工逐题编写。默认生成 2～4 个错误：

```text
bug_count=2：默认 60% 的多错误尝试预算
bug_count=3：默认 30%
bug_count=4：默认 10%
```

这些比例用于安排 mutation 尝试顺序；最终 accepted 多错误数据也按第 6.7 节的 60%/30%/10% 配额采样。某类有效率较低时应增加该类尝试量，而不是降低验证标准。

组合算法：

1. 枚举 reference 上所有受支持的单点编辑候选；
2. 按 seed 随机选定目标 `bug_count`；
3. 构建兼容图：两个编辑不重叠、不修改同一 AST 节点、不互相撤销，且应用顺序不会改变另一个编辑的定位；
4. 从兼容图采样 edit set，优先覆盖不同 family 和不同代码区域；
5. 一次性从原始 AST 应用全部编辑，不能在反复 unparse 后继续按旧行号编辑；
6. 验证最终 AST 的语义编辑数恰好等于 `bug_count`；
7. 编译并运行完整 grader；
8. 对组合中的每个 edit 单独构造 single mutant 并运行 grader，确认它本身是可观测的真实错误；
9. 对组合 mutant 重放，检查确定性；
10. 通过反事实流程决定其行为数据，绝不能预设“多 bug 一定需要执行”。

每个组合必须满足：

```text
2 <= semantic_edit_count <= configured_max_bug_count
每个单独 edit 都使 reference 失去 100% 通过
组合 mutant 失去 100% 通过
所有编辑 pairwise compatible
最终程序可解析、可编译
组合失败可稳定重放
```

默认优先保留 `0 < pass_rate < 1` 的组合，因为它们能产生有区分度的 failing case。`pass_rate=0` 的组合只有在运行稳定、存在具体 failing input、且反事实修复可成功时才可保留，并应单独标记 `all_tests_failed=true`。

### 6.3 防止多错误互相遮蔽

仅验证“组合代码失败”还不够。需要执行以下检查：

- `individual_harmfulness`：每个 edit 单独应用时都导致至少一个测试失败；
- `no_cancellation`：组合后的 AST 不能恢复任一 edit 修改前的语义；
- `survivor_check`：修复任意一个 edit、保留其他 edits 时，程序仍不应 100% 通过；
- `location_separation`：默认要求错误位于不同 statement；如位于同一 statement，必须由 operator 明确声明兼容；
- `family_diversity`：优先不同 family，但不作为硬性接受条件；
- `failure_signature`：记录各 single mutant 与组合 mutant 的失败签名，供分析遮蔽程度；
- `repair_completeness`：最终正确代码必须消除所有已记录的语义 edits，而不只是修复 grader 暴露的第一个错误。

如果两个 edit 组合后导致其中一个不可执行、不可定位或完全被另一个 edit 覆盖，标记 `masked_mutation` 并丢弃该组合。

### 6.4 AST 编辑证明

每个 edit 返回结构化记录。多点错误保存有序数组，而不是把多个错误压成一个字符串：

```json
{
  "bug_count": 2,
  "edits": [
    {
      "operator": "compare_boundary",
      "node_type": "Gt",
      "source_span": {"line": 12, "col": 15, "end_line": 12, "end_col": 16},
      "before": ">",
      "after": ">=",
      "semantic_edit_count": 1
    },
    {
      "operator": "range_stop_offset",
      "node_type": "Call",
      "source_span": {"line": 20, "col": 13, "end_line": 20, "end_col": 21},
      "before": "range(n)",
      "after": "range(n - 1)",
      "semantic_edit_count": 1
    }
  ],
  "semantic_edit_count": 2
}
```

这些记录只进入内部 metadata，永不进入 prompt/messages。验证器必须从原始 reference 重新 parse 并重新应用 edit set；AST unparse 引起的格式变化不计为额外语义编辑。

### 6.5 模型自然错误和自然正确解

对 SFT synthesis pool 中每个题目，让基础模型仅根据公开题面和 starter code 独立生成若干完整候选解。生成 prompt 不包含 reference、tests、expected outputs 或 mutation 信息。

每个候选通过内部 grader 后分流：

```text
两次 100% 通过
-> model_natural_correct
-> 构造“已有可靠解时直接提交”数据

稳定失败且能提取完整代码
-> model_natural_failure
-> 构造失败状态并进入同一反事实流程

输出无法解析、严重越权或执行不稳定
-> discarded
```

自然错误可能同时包含多个问题，因此：

- `bug_count=null`，除非有独立静态分析能可靠证明；
- 不伪造 AST edit record；
- 不要求与 reference 只有局部差异；
- 可以包含 compile error、runtime error、算法错误、复杂度错误和 I/O 错误；
- 如果 `submit` 无法提供具体 failing input，则不能生成“失败样例重放”，但仍可尝试“直接修复”；
- 生成模型、revision、prompt hash、seed 和原始输出必须保存；
- 同题的自然候选必须按 normalized code 去重。

为了降低 policy/style 偏移，默认同时从开发模型和正式基础模型采集候选；metadata 用 `generator_model` 区分。不可使用 APPS test 题采集自然错误。

### 6.6 候选筛选与去重

建议进入反事实候选池的默认上限为：每题最多 3 个单点 synthetic candidates、3 个多点 synthetic candidates、3 个模型自然失败和 2 个模型自然正确解。最终写入训练集时还必须遵守第 6.7 节更严格的单题上限。

采用以下去重键：

```text
normalized_code_hash
id + ordered_mutation_set_hash
id + pass_rate + normalized_failing_input + exception_type
id + model_name + generation_seed + normalized_code_hash
```

同题、同 failure signature 的候选优先保留：来源更自然者、反事实 utility 更高者、执行更稳定者、与已有代码差异更大者。错误来源不得作为行为标签；无论 single、multi 还是 natural，都要重新测量工具是否有用。

### 6.7 最终数据规模与比例

默认正式数据集包含 **1200 条通过全部验收的 episode**。一条 episode 对应 `sft_messages.jsonl` 中的一行，可以包含多个 assistant/tool 回合。1200 是默认可执行目标，位于此前规划的 800～1500 条范围内；CLI 可通过 `--target-count` 修改，但必须按同一比例重新计算整数配额。

#### 候选生成预算

候选“尝试数”不是训练数据数。生成器可以产生大量候选，只有满足执行、反事实、泄漏和重放检查的 episode 才计入最终 1200 条。

| 候选来源 | 每题最大尝试/候选数 | 1000 题理论上限 | 何时停止 |
|---|---:|---:|---|
| 单点 AST 错误 | 3 | 3000 | 该来源接受配额填满 |
| 2～4 点 AST 组合错误 | 3 | 3000 | 该来源接受配额填满 |
| 模型自然候选 | 5 | 5000 | 自然错误和自然正确解配额均填满，或候选池耗尽 |
| verified reference | 每题至多 1 | 1000 | reference 直接提交配额填满 |

生成顺序应按 id、difficulty 和 seed 轮转，不能先把少数题的上限全部用完。达到某一来源的 accepted 配额后停止为该来源继续消耗模型推理；未通过样本继续写入 reject statistics，但不进入 SFT。

#### 最终 1200 条的来源比例

| 最终代码状态/候选来源 | 数量 | 比例 |
|---|---:|---:|
| 单点 AST 错误产生的修复 episode | 360 | 30% |
| 多点 AST 错误产生的修复 episode | 360 | 30% |
| 模型自然错误产生的修复 episode | 300 | 25% |
| verified reference 直接提交 | 120 | 10% |
| 模型自然正确解直接提交 | 60 | 5% |
| **合计** | **1200** | **100%** |

这样，85% 的数据教授调试和修复，15% 教授在已有可靠解时避免无效执行；人工可控错误与自然错误同时存在，且多点错误不会沦为少量附录。

#### 最终 1200 条的行为比例

| 模型需要学习的行为 | 数量 | 比例 |
|---|---:|---:|
| 得到 submit 反馈后直接修复 | 300 | 25% |
| 重放 submit 返回的失败样例后修复 | 360 | 30% |
| 首次 submit 前主动设计并执行测试 | 360 | 30% |
| 已有可靠解时直接 submit | 180 | 15% |
| **合计** | **1200** | **100%** |

#### 来源与行为的完整交叉表

| 候选来源 | 提交后直接修复 | 重放失败样例 | 提交前主动测试 | 可靠解直接提交 | 合计 |
|---|---:|---:|---:|---:|---:|
| 单点 AST 错误 | 120 | 120 | 120 | 0 | 360 |
| 多点 AST 错误 | 60 | 150 | 150 | 0 | 360 |
| 模型自然错误 | 120 | 90 | 90 | 0 | 300 |
| verified reference | 0 | 0 | 0 | 120 | 120 |
| 模型自然正确解 | 0 | 0 | 0 | 60 | 60 |
| **合计** | **300** | **360** | **360** | **180** | **1200** |

该表是 accepted data 的采样目标，不是标签生成规则。例如某个多点错误只有在反事实实验确实证明执行有帮助时，才能进入“重放失败样例”或“提交前主动测试”列。若某个单元格不足，处理顺序是：增加不同 problem 的候选生成、增加对应来源的模型采样、继续运行严格反事实筛选；仍不足则输出 shortage report 并停止验收，不得将其他行为重标后填入。

#### 多点错误内部比例

多点错误共 360 条，默认分配为：

| AST 错误数 | 数量 | 占多点错误比例 | 占完整数据比例 |
|---|---:|---:|---:|
| 2 个错误 | 216 | 60% | 18% |
| 3 个错误 | 108 | 30% | 9% |
| 4 个错误 | 36 | 10% | 3% |
| **合计** | **360** | **100%** | **30%** |

若某个 bug count 的有效率过低，只能增加尝试预算或在报告中说明短缺；不能用重复样本补齐。

#### 难度比例

默认对最终 1200 条做 difficulty-stratified sampling：

| APPS 难度 | 数量 | 比例 |
|---|---:|---:|
| `introductory` | 300 | 25% |
| `interview` | 540 | 45% |
| `competition` | 360 | 30% |
| **合计** | **1200** | **100%** |

每种候选来源和每种行为都应尽量覆盖三个难度。配额分配使用 largest-remainder method，避免四舍五入后总数不等于 1200。若 cleaned SFT pool 某难度供给不足，必须报告原始可用量、缺口和实际分布，正式验收判定失败；不得从 RL/dev pool 借题。

#### 单题上限与去重后的计数单位

- 同一 problem 最多进入 6 条最终 episode；
- 同一 problem × 同一行为最多 2 条；
- 同一 problem × 同一候选来源最多 3 条；
- 同一 normalized candidate code 最多进入 1 条；
- 同一 failure signature 在同题内最多进入 2 条，且候选来源或行为必须不同；
- 数量统计发生在去重、最终重放和全部 QA 之后；
- rejected、ambiguous、quarantine 和反事实对照分支不计入 1200 条。

#### 当目标总量不是 1200 时

`--target-count N` 使用上述比例计算配额：

```python
raw_quota_i = N * ratio_i
quota = floor(raw_quota_i)
将剩余名额按小数部分从大到小依次补 1
```

来源表、行为表和难度表分别计算，并由一个约束采样器满足交叉条件。若 `N` 太小导致某些单元格为 0，应至少保证所有候选来源和四种行为各有 1 条；正式训练不建议 `N < 200`。

## 7. 反事实生成与公平性

### 7.1 分支输入

每个分支的 system prompt、题面、candidate、starter code、I/O 说明、temperature、top-p、max tokens 和模型版本必须一致；唯一允许变化的是实验处理：是否提供 `run_candidate` observation。

建议为每个状态预先生成 `k` 个 seed，并在 direct/run 分支中配对使用。metadata 中保存 `paired_seed`，便于计算配对增益。不要把 direct 分支中生成的错误 repair 作为 run 分支的新起点。

### 7.2 成功定义

一次 repair attempt 只有在以下条件全部满足时记为成功：

1. 模型输出可解析为合法的 `submit(code)`；
2. `code` 是完整程序，不是 diff 或代码片段；
3. grader 在全部 cleaned tests 上通过；
4. 第二次 replay 仍通过；
5. 未触发资源或安全违规。

任何自然语言自称“修复完成”均不算成功。

### 7.3 可观察 reasoning

若启用短 rationale，其生成器只能读取：题面、starter code、当前 candidate、已公开的 submit feedback、已公开的 run observation。禁止读取 reference、expected outputs、mutation metadata 和 grader 内部日志。QA 应扫描明显泄漏短语，但自动扫描不能替代 prompt provenance 审计。

默认推荐 `rationale_mode=none`：assistant 直接输出合法 tool call，以减少不可验证推理和 oracle 泄漏。

## 8. 轨迹与 SFT 导出

### 8.1 规范化消息

导出采用 JSONL，每行一个 episode。建议保留标准 `messages` 和显式的逐消息训练标记：

```json
{
  "id": "apps-123-post-submit-failure-replay-abc123",
  "tools": ["<run_candidate schema>", "<submit schema>"],
  "messages": [
    {"role": "system", "content": "...", "trainable": false},
    {"role": "user", "content": "Problem... Current candidate... Previous grader feedback...", "trainable": false},
    {"role": "assistant", "tool_calls": [{"name": "run_candidate", "arguments": {"input": "..."}}], "trainable": true},
    {"role": "tool", "name": "run_candidate", "content": "{...}", "trainable": false},
    {"role": "assistant", "tool_calls": [{"name": "submit", "arguments": {"code": "..."}}], "trainable": true},
    {"role": "tool", "name": "submit", "content": "{\"status\":\"accepted\",...}", "trainable": false}
  ],
  "metadata": {"behavior_sequence": ["post_submit_failure_replay", "post_submit_direct_repair"]}
}
```

实际 tool call 序列化应调用所选 tokenizer 的官方 chat template，不手写模型专用控制 token。`tools` 保存完整 JSON schemas，而不是字符串占位符。

### 8.2 哪些回合计算 loss

| 行为 | 上下文（全部 mask） | 训练目标（计算 loss） |
|---|---|---|
| 提交失败后直接修复 | problem、buggy candidate、历史 submit feedback | `submit(repaired_code)` |
| 提交失败后重放失败样例 | problem、buggy candidate、历史 submit feedback；中间 tool response | `run_candidate(failing_input)`；最终正确的 `submit(repaired_code)` |
| 首次提交前主动设计测试 | problem、candidate；中间 tool response | `run_candidate(model_proposed_input)`；最终正确的 `submit(repaired_or_unchanged_code)` |
| 已有可靠解时直接提交 | problem、starter code | `submit(verified_code)` |

历史错误提交不应重建为 trainable assistant 回合。推荐将其投影为 user state 中的 `Current candidate` 和 `Previous grader feedback`。若为了兼容多轮框架必须保留历史 assistant/tool 消息，则历史错误 assistant 回合必须 `trainable=false`。多点错误产生的中间失败修复同样必须 mask，或按第 5.6 节切成状态级样本。

### 8.3 token-level loss mask

使用 tokenizer chat template 得到 token 后构造 `labels`：

```python
labels = [-100] * len(input_ids)
for span in assistant_spans:
    if span.message.trainable:
        labels[span.start:span.end] = input_ids[span.start:span.end]
```

必须 mask：system、user、tool schema 展开、tool response、padding、错误 candidate，以及所有 `trainable=false` assistant 历史。默认对 trainable assistant 的完整 tool-call token（含 name、arguments 和代码）计算 loss。

#### 为什么 tool response 不计算 loss，模型仍然能学会使用工具

需要严格区分两个回合：

```text
assistant: run_candidate(input=...)   <- 模型产生的工具调用，计算 loss
tool:      stdout/stderr/error         <- 环境产生的观察，不计算 loss
```

模型需要学习的是：

1. 在什么状态下选择 `run_candidate`；
2. 如何生成合法的工具名和 `input` 参数；
3. 看到工具返回后，如何生成下一次 `run_candidate` 或 `submit(code)`。

这三项都有直接监督：assistant 的工具调用 token 计算 loss；工具返回之后的 assistant 动作也计算 loss。工具返回本身是环境数据，不应训练模型生成，否则模型可能学会伪造 stdout 或错误信息。

tool response 的 `labels=-100` 只表示“不要求模型预测这些 token”，不能把它从上下文中删除。除 padding 外，它的 `attention_mask` 必须仍为 1：

```text
位置                  labels                 attention_mask
system                -100                   1
user state            -100                   1
assistant tool call   对应 input_ids          1
tool response         -100                   1
assistant submit      对应 input_ids          1
padding               -100                   0
```

因此，在预测后续 `submit(code)` 时，模型仍能通过 self-attention 读取完整 stdout/stderr/error。后续 assistant token 的 loss 会训练模型根据这段 observation 选择和生成正确动作。换言之：tool response 没有“自身的 next-token prediction loss”，但它仍是产生后续监督答案的条件上下文。

错误实现是同时设置：

```text
tool response labels = -100
tool response attention_mask = 0
```

这会让后续 assistant 看不到工具结果，确实无法学会利用 observation。正确实现只能 mask labels，不能 mask 非 padding 上下文的 attention。

导出和单元测试必须分别断言：

- assistant 的 `run_candidate` 工具名、JSON 参数和控制 token 全部位于 supervised span；
- role=`tool` 的 response token 全部 `labels=-100`；
- role=`tool` 的 response token 全部 `attention_mask=1`；
- observation 后最终正确 `submit(code)` 的完整 assistant span 被监督；
- 中间失败的 assistant `submit(code)` 即使 role 是 assistant，也必须 `labels=-100`；
- 解码 `labels != -100` 后能够看到完整工具调用和最终正确提交，而看不到任何环境返回。

导出验证器必须解码所有 `labels != -100` 的 token，并检查其中不存在 seed buggy code 的完整匹配。若 repaired code 与 buggy code 大部分相同，这是正常的；检查重点是不能把“历史错误提交回合”整体设为 target。

### 8.4 推荐文件

- `episodes.jsonl`：完整可重放、带内部引用 ID 的合成轨迹；
- `sft_messages.jsonl`：去除秘密字段后的训练消息；
- `sft_tokenized/`：可选的 `input_ids/attention_mask/labels`；
- `metadata.jsonl`：不送入模型的审计元数据；
- `dataset_manifest.json`：目标数量、实际数量、来源/行为/难度/bug count 交叉统计与缺口；
- `qa_report.json` 与 `qa_report.md`：机器可读和人类可读报告。

`dataset_manifest.json` 至少包含：

```json
{
  "target_episodes": 1200,
  "accepted_episodes": 1200,
  "sft_jsonl_rows": 1200,
  "attempted_candidates": 0,
  "valid_before_sampling": 0,
  "rejected_candidates": 0,
  "by_candidate_origin": {},
  "by_behavior": {},
  "origin_by_behavior": {},
  "by_difficulty": {},
  "by_bug_count": {},
  "quota_shortfalls": []
}
```

所有 count 从最终文件重新扫描计算，不能只相信生成进程的内存计数器。

## 9. Metadata schema

metadata 不得拼接回 prompt。建议 schema 如下；实现时使用 Pydantic/JSON Schema 严格校验：

```json
{
  "schema_version": "1.0",
  "episode_id": "apps-123-post-submit-failure-replay-abc123",
  "id": "123",
  "dataset": "codeparrot/apps",
  "dataset_split": "train",
  "pool": "sft",
  "difficulty": "interview",
  "io_mode": "stdin",
  "question_hash": "sha256:...",
  "starter_code_hash": "sha256:...",

  "reference": {
    "solution_index": 0,
    "code_hash": "sha256:...",
    "verified": true,
    "verification_pass_rate": 1.0,
    "replay_count": 2
  },

  "candidate": {
    "origin": "synthetic_single | synthetic_multi | model_natural_failure | verified_reference | model_natural_correct",
    "code_hash": "sha256:...",
    "is_buggy_context": true,
    "generator_model": null,
    "generator_revision": null,
    "generation_seed": null,
    "generation_prompt_hash": null
  },

  "mutation": {
    "is_mutated": true,
    "bug_count": 2,
    "semantic_edit_count": 2,
    "all_individually_harmful": true,
    "survivor_check_passed": true,
    "masked_mutation": false,
    "edits": [
      {
        "operator": "compare_boundary",
        "family": "comparator",
        "node_type": "Gt",
        "source_span": {"line": 12, "col": 15, "end_line": 12, "end_col": 16},
        "before": ">",
        "after": ">="
      },
      {
        "operator": "range_stop_offset",
        "family": "range_bound",
        "node_type": "Call",
        "source_span": {"line": 20, "col": 13, "end_line": 20, "end_col": 21},
        "before": "range(n)",
        "after": "range(n - 1)"
      }
    ]
  },

  "seed_submit": {
    "status": "wrong_answer",
    "passed": 7,
    "total": 10,
    "pass_rate": 0.7,
    "failing_input_hash": "sha256:...",
    "failure_signature": "sha256:..."
  },

  "counterfactual": {
    "k": 3,
    "without_run_successes": 0,
    "without_run_rate": 0.0,
    "with_run_successes": 3,
    "with_run_rate": 1.0,
    "utility": 1.0,
    "threshold_high": 0.6666667,
    "threshold_low": 0.3333333,
    "min_utility": 0.3333333,
    "paired_seeds": [11, 22, 33]
  },

  "execution_query": {
    "kind": "none | submit_failing_case | model_proposed",
    "input_hash": "sha256:...",
    "proposal_model": null,
    "proposal_prompt_hash": null,
    "proposal_raw_hash": null,
    "observable_fields": ["question", "starter_code", "candidate"],
    "matches_submit_failing_input": true
  },

  "behavior_sequence": [
    "post_submit_failure_replay",
    "post_submit_direct_repair"
  ],
  "decision_steps": [
    {
      "step": 0,
      "behavior": "post_submit_failure_replay",
      "assistant_trainable": true,
      "downstream_success": true
    },
    {
      "step": 1,
      "behavior": "post_submit_direct_repair",
      "assistant_trainable": true,
      "downstream_success": true
    }
  ],
  "solution_origin": "verified_reference | model_repair | model_natural_correct",
  "final_code_hash": "sha256:...",
  "final_pass_rate": 1.0,
  "final_replay_passed": true,
  "tool_sequence": ["run_candidate", "submit"],
  "tool_call_count": 2,

  "model": {
    "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "revision": "pinned-revision",
    "template_hash": "sha256:...",
    "generation_config_hash": "sha256:..."
  },

  "runtime": {
    "python_version": "3.x.y",
    "grader_version": "git:...",
    "sandbox_config_hash": "sha256:..."
  },

  "quality": {
    "schema_valid": true,
    "replay_valid": true,
    "leakage_scan_passed": true,
    "loss_mask_valid": true,
    "duplicate_of": null,
    "reject_reasons": []
  },

  "created_at": "ISO-8601 UTC timestamp"
}
```

对于自然错误，`mutation` 必须是 `null`，不得编造 `bug_count` 或 edit records。对于单点错误，`bug_count=1` 且 `edits` 长度为 1。对于直接提交轨迹，`counterfactual` 和 `execution_query` 可以为 `null`。

原始 code、failing input、prompts 和 observations 可保存在受控 artifact store，以 content hash 在 metadata 中引用。公开数据若需要完整可重放，应将可公开字段写入 `episodes.jsonl`；hidden expected outputs 永不写入 SFT 文件。

## 10. 数据质量检查

### 10.1 Schema 与协议

- 所有 JSONL 行可解析并通过 schema；
- 工具名只能是 `run_candidate` 或 `submit`；
- 参数键严格分别为 `{input}`、`{code}`；
- 所有 submit 参数都是完整非空代码；
- “失败样例重放”的 run input 与公开 failing input 完全一致；
- “首次提交前主动验证”在第一次 run 前没有 submit feedback；
- “已有可靠解时直接提交”不包含 run；
- “提交失败后直接修复”目标不包含 run。

### 10.2 正确性与可重放

- verified reference 两次通过；
- mutant 两次呈现同一 failure signature；
- 所有保留轨迹最终两次 100% 通过；
- tool observations 可由存档代码、输入和 pinned runtime 重放；
- `passed/total/pass_rate` 一致；
- failing input 必须确实在 grader 中失败。

### 10.3 反事实有效性

- “直接修复”满足 `p_without_run >= HIGH`；
- 两种需要执行的行为均满足 without-run 低、with-run 高且 utility 达标；
- 可靠 reference 或模型自然正确解可直接提交通过；
- 所有采样次数、seed 和分支输出齐全；
- 不用 mutation family 替代反事实测量；
- ambiguous 不进入训练集。

### 10.4 泄漏检查

对所有模型可见消息执行：

- expected output 与 reference 片段精确/模糊匹配扫描；
- mutation operator、source span、before/after 等内部字段名扫描；
- 主动验证 prompt 字段白名单检查，而非仅黑名单；
- submit response schema 快照测试，确保没有 `expected`、`actual`、`test_index`；
- tool response 和内部 grader result 做字段级 diff；
- 人工抽样确认 reasoning 没有依赖不可观察信息。

注意：题面本身可能包含示例输出，不能把所有输出文本一概视为泄漏。泄漏检查应区分公开题面示例与 hidden expected outputs。

### 10.5 Loss mask 检查

- 每条样本至少有一个 supervised token；
- 非 trainable role 的 labels 全为 `-100`；
- 除 padding 外，system/user/tool/assistant 上下文的 `attention_mask` 均为 1；
- 错误 candidate 所在消息全 mask；
- “失败样例重放”和“主动设计测试”中的有效 run 动作以及最终正确 submit 均 supervised；
- “直接修复”和“可靠解直接提交”的正确 submit supervised；
- tool observations 全 mask；
- 通过 decode supervised spans 做人工可读快照。

### 10.6 去重与污染

- id 不跨 SFT/RL/dev；
- APPS test ID 不出现；
- episode_id、normalized code hash 不重复；
- 同题同 failure signature 的近重复受限；
- train/dev 导出按 id group split，不能按 trajectory 随机切分；
- 报告每题轨迹数上限，防止少量题主导训练。

### 10.7 分布报告

报告必须同时展示 `attempted / valid-before-sampling / accepted-after-sampling / rejected`，并验证第 6.7 节的最终配额。比例通过对已验证样本采样实现，不允许通过重标或降低质量门槛实现。

至少报告：

- 四种行为以及 ambiguous/discarded 数量；
- difficulty、I/O mode、candidate origin、bug count、mutation family 分布；
- 每题 mutant 数、接受率；
- `p_without_run`、`p_with_run`、utility 分布；
- pass_rate 与 failure 类型分布；
- 工具调用数、输入长度、代码 token 长度；
- reject reason 排名；
- 主动验证 proposal 的可解析率和有效增益率。

## 11. 推荐目录结构

```text
coding-agent-synthesis/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   ├── gate_200.yaml
│   └── tool_schemas.json
├── data/
│   ├── raw/apps/
│   ├── cleaned/
│   ├── splits/
│   ├── artifacts/
│   ├── episodes/
│   ├── sft/
│   └── reports/
├── src/synthesis/
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── apps_loader.py
│   ├── normalize.py
│   ├── splitter.py
│   ├── sandbox.py
│   ├── harness.py
│   ├── grader.py
│   ├── tools.py
│   ├── feedback_projection.py
│   ├── mutation/
│   │   ├── base.py
│   │   ├── operators.py
│   │   ├── compose.py
│   │   └── validate.py
│   ├── natural_candidates.py
│   ├── prompting/
│   │   ├── state_builder.py
│   │   ├── repair.py
│   │   └── propose_input.py
│   ├── generation.py
│   ├── counterfactual.py
│   ├── classify.py
│   ├── trajectory.py
│   ├── export_sft.py
│   ├── loss_mask.py
│   ├── dedup.py
│   ├── qa.py
│   ├── showcase.py
│   └── report.py
├── tests/
│   ├── test_apps_loader.py
│   ├── test_harness.py
│   ├── test_tool_contract.py
│   ├── test_mutation_single_edit.py
│   ├── test_counterfactual_rules.py
│   ├── test_active_validation_prompt_whitelist.py
│   ├── test_loss_mask.py
│   ├── test_leakage.py
│   └── fixtures/
└── scripts/
    └── run_gate_200.sh
```

## 12. 脚本模块职责

| 模块 | 职责 |
|---|---|
| `apps_loader.py` | 流式读取 JSONL，二次解析 JSON string 字段，输出原始 schema |
| `normalize.py` | I/O mode、输入字符串、输出比较规则和代码规范化 |
| `splitter.py` | 固定 seed、difficulty-stratified、problem-level 互斥划分 |
| `sandbox.py` | 资源隔离、超时、输出截断、执行审计 |
| `harness.py` | stdin 与 function-call 两类 APPS adapter |
| `grader.py` | 私有 tests 执行和完整内部结果 |
| `tools.py` | 对模型暴露的两个严格工具及 schema validation |
| `feedback_projection.py` | 从内部 grader result 投影出 pass rate + failing input |
| `mutation/*` | 枚举单点 AST edits、构建兼容图、自动组合多点错误、验证 edit count 和遮蔽 |
| `natural_candidates.py` | 仅从公开题面生成模型自然候选，并按 grader 结果分流正确解和自然错误 |
| `prompting/state_builder.py` | 只用 observable 字段构造四种行为状态和多轮状态切片 |
| `prompting/propose_input.py` | 提交前主动验证的白名单 prompt，解析模型自主输入 |
| `generation.py` | pinned 模型与生成参数，保存原始输出 |
| `counterfactual.py` | 配对 direct/run 分支、多次采样、真实 grader 验证 |
| `classify.py` | 仅根据统计阈值和协议约束分类 |
| `trajectory.py` | 选取成功分支，生成 canonical messages |
| `export_sft.py` | 秘密字段剥离、JSONL 导出、chat template 渲染 |
| `loss_mask.py` | message span 到 token label 的精确 mask |
| `dedup.py` | problem/code/failure signature 多层去重 |
| `qa.py` | schema、replay、反事实、泄漏、mask、split 检查 |
| `showcase.py` | 从真实 artifacts 自动生成脱敏的 `SYNTHESIS_SHOWCASE.md`，逐步展示原始题、候选、工具消息、loss mask 和 metadata |
| `report.py` | 门测与正式运行的分布、成本、拒绝原因报告 |

## 13. 核心伪代码

### 13.1 清洗和 reference 验证

```python
def clean_problem(raw):
    rec = parse_apps_record_strict(raw)
    io_spec = normalize_io(rec.input_output, rec.starter_code)
    verified = []

    for idx, code in enumerate(rec.solutions):
        if not parses_and_compiles(code):
            continue
        r1 = private_grader.grade(code, io_spec)
        r2 = private_grader.grade(code, io_spec)
        if r1.accepted and r2.accepted and same_result(r1, r2):
            verified.append((idx, code, r1))

    if not verified:
        return reject(rec.id, "no_verified_reference")

    chosen = min(verified, key=lambda x: ast_node_count(x[1]))
    return CleanProblem(rec, io_spec, chosen, verified)
```

### 13.2 单点错误生成

```python
def build_single_mutants(problem, cfg):
    accepted = []
    for edit in enumerate_supported_single_edits(problem.reference_code):
        mutant, edit_record = apply_exactly_one_edit(problem.reference_code, edit)
        if not validate_single_ast_edit(problem.reference_code, mutant, edit_record):
            continue
        if not parses_and_compiles(mutant):
            continue

        s1 = private_grader.grade(mutant, problem.io_spec)
        s2 = private_grader.grade(mutant, problem.io_spec)
        if not (0.0 < s1.pass_rate < 1.0):
            continue
        if failure_signature(s1) != failure_signature(s2):
            continue
        if is_duplicate(problem.id, mutant, s1, accepted):
            continue

        accepted.append(Mutant(mutant, edit_record, s1))
        if len(accepted) == cfg.max_single_mutants_per_problem:
            break
    return accepted
```

### 13.3 多点错误自动组合

```python
def build_multi_mutants(problem, cfg):
    edits = enumerate_supported_single_edits(problem.reference_code)
    graph = build_compatibility_graph(edits)
    accepted = []

    for bug_count in sample_bug_counts(cfg.multi_bug_distribution):
        for edit_set in sample_compatible_sets(graph, size=bug_count):
            mutant, records = apply_edits_from_original_ast(
                problem.reference_code, edit_set
            )
            if not validate_exact_edit_count(
                problem.reference_code, mutant, expected=bug_count
            ):
                continue
            if not parses_and_compiles(mutant):
                continue

            individual = [
                private_grader.grade(
                    apply_edits_from_original_ast(problem.reference_code, [e])[0],
                    problem.io_spec,
                )
                for e in edit_set
            ]
            if not all(not r.accepted for r in individual):
                continue
            if not survivor_check(problem, edit_set):
                continue

            combined_1 = private_grader.grade(mutant, problem.io_spec)
            combined_2 = private_grader.grade(mutant, problem.io_spec)
            if combined_1.accepted or not same_failure(combined_1, combined_2):
                continue
            if is_masked(edit_set, individual, combined_1):
                continue

            accepted.append(MultiMutant(mutant, records, combined_1))
            if len(accepted) == cfg.max_multi_mutants_per_problem:
                return accepted
    return accepted
```

### 13.4 模型自然候选采集

```python
def harvest_natural_candidates(problem, model, seeds, cfg):
    state = build_problem_only_state(problem.public_fields)
    outputs = [model.generate_complete_solution(state, seed=s) for s in seeds]
    results = []

    for output in deduplicate_by_normalized_code(outputs):
        code = parse_complete_code(output)
        if code is None:
            results.append(discard(output, "unparseable_code"))
            continue

        r1 = private_grader.grade(code, problem.io_spec)
        r2 = private_grader.grade(code, problem.io_spec)
        if not same_result(r1, r2):
            results.append(quarantine(code, "nondeterministic"))
        elif r1.accepted:
            results.append(NaturalCorrect(code, r1))
        else:
            results.append(NaturalFailure(code, r1))
    return results
```

### 13.5 提交后状态的反事实判定

```python
def classify_post_submit(problem, candidate, seeds, cfg):
    public_feedback = project_submit_feedback(candidate.seed_submit)
    state = build_post_submit_state(
        problem=problem.public_fields,
        candidate=candidate.code,
        feedback=public_feedback,
    )

    direct = [
        generate_repair_then_submit(state, seed=s)
        for s in seeds
    ]
    direct_ok = [verify_final_submit(x) for x in direct]
    p_without_run = mean(direct_ok)

    if p_without_run >= cfg.high:
        return accept_behavior(
            "post_submit_direct_repair",
            select_replayable_success(direct),
        )

    failing = public_feedback.failing_input
    if failing is None:
        return discard("failure_replay_requires_failing_input")

    obs = run_candidate(current_candidate=candidate.code, input=failing)
    run_state = append_tool_observation(state, "run_candidate", failing, obs)
    with_replay = [
        generate_repair_then_submit(run_state, seed=s)
        for s in seeds
    ]
    replay_ok = [verify_final_submit(x) for x in with_replay]
    p_with_run = mean(replay_ok)

    if p_without_run <= cfg.low and p_with_run >= cfg.high \
            and p_with_run - p_without_run >= cfg.min_utility:
        return accept_behavior(
            "post_submit_failure_replay",
            select_replayable_success(with_replay),
            observation=obs,
        )
    return ambiguous_or_discard(direct_ok, replay_ok)
```

### 13.6 首次提交前主动验证的反事实判定

```python
def classify_pre_submit(problem, candidate, seeds, cfg):
    # Builder 使用严格白名单，不能接收 grader_result/mutation/reference 参数。
    state = build_pre_submit_state(
        problem=problem.public_fields,
        candidate=candidate.code,
    )

    no_run = [
        generate_direct_submit_or_repair(state, seed=s)
        for s in seeds
    ]
    no_run_ok = [verify_final_submit(x) for x in no_run]
    p_no_run = mean(no_run_ok)

    proposal_raw = propose_diagnostic_input(state, seed=cfg.proposal_seed)
    proposed_input = parse_single_input(proposal_raw, problem.io_spec)
    assert proposal_provenance_is_observable_only()

    obs = run_candidate(current_candidate=candidate.code, input=proposed_input)
    run_state = append_tool_observation(state, "run_candidate", proposed_input, obs)
    with_run = [
        generate_repair_then_submit(run_state, seed=s)
        for s in seeds
    ]
    with_run_ok = [verify_final_submit(x) for x in with_run]
    p_run = mean(with_run_ok)

    if p_no_run <= cfg.low and p_run >= cfg.high \
            and p_run - p_no_run >= cfg.min_utility:
        return accept_behavior(
            "pre_submit_active_validation",
            trajectory=select_replayable_success(with_run),
            proposal_raw=proposal_raw,
            proposed_input=proposed_input,
            observation=obs,
        )
    return ambiguous_or_discard(no_run_ok, with_run_ok)
```

### 13.7 可靠解直接提交

```python
def build_direct_submission(problem, verified_solution):
    code = verified_solution.code
    assert private_grader.grade(code, problem.io_spec).accepted
    assert private_grader.grade(code, problem.io_spec).accepted
    return trajectory(
        context=build_problem_only_state(problem.public_fields),
        target=tool_call("submit", {"code": code}),
        solution_origin=verified_solution.origin,
    )
```

### 13.8 多轮 episode 的 SFT 导出

```python
def export_episode(episode, tokenizer):
    assert episode.final_submit.accepted
    public = strip_internal_fields(episode)
    validate_no_oracle_fields(public)

    messages = build_canonical_messages(public)
    input_ids, spans = apply_chat_template_with_spans(
        tokenizer, messages, tools=STRICT_TOOL_SCHEMAS
    )
    labels = mask_all_tokens(input_ids)
    for span in spans:
        if span.role == "assistant" and span.trainable:
            labels[span.start:span.end] = input_ids[span.start:span.end]

    validate_loss_mask(messages, input_ids, labels, spans)
    validate_decision_slices_for_qa(messages, spans)
    return {
        "messages": messages,
        "input_ids": input_ids,
        "labels": labels,
    }
```

## 14. CLI 设计

建议提供单一入口 `python -m synthesis.cli`，子命令如下：

```text
prepare-apps     解析、清洗、reference 验证、problem-level 划分
generate-candidates  生成单点错误、多点错误和模型自然候选
synthesize       运行四种行为的轨迹合成与反事实分类
export-sft       导出 messages 与可选 tokenized labels
qa               执行全量质量检查和重放
gate-200         从头运行首批 200 条门测并生成报告
showcase         从已验收 episode 生成可阅读的具体数据展示
report           重新汇总已有 artifacts
```

关键参数：

```text
--apps-path PATH
--output-dir PATH
--split-seed INT                 default: 42
--generation-seed INT            default: 2026
--model NAME
--model-revision REV             必须 pin；不得正式运行 latest 浮动版本
--dtype {bf16,fp16}
--tensor-parallel-size INT
--temperature FLOAT
--top-p FLOAT
--max-new-tokens INT
--counterfactual-samples INT      default: 3
--high-threshold FLOAT            default: 0.6666667
--low-threshold FLOAT             default: 0.3333333
--min-utility FLOAT               default: 0.3333333
--candidate-sources CSV           default: synthetic_single,synthetic_multi,model_natural
--max-single-mutants-per-problem INT  default: 3
--max-multi-mutants-per-problem INT   default: 3
--max-natural-failures-per-problem INT default: 3
--natural-generations-per-problem INT  default: 5
--multi-bug-counts CSV            default: 2,3,4
--multi-bug-weights CSV           default: 0.6,0.3,0.1
--mutation-families CSV
--max-actions-per-episode INT      default: 6
--max-submit-calls INT             default: 3
--max-run-calls INT                default: 3
--candidate-timeout-sec FLOAT
--candidate-memory-mb INT
--max-output-bytes INT
--max-input-bytes INT
--max-code-tokens INT
--rationale-mode {none,short}     default: none
--target-count INT                 default: 1200
--source-ratios CSV                default: 0.30,0.30,0.25,0.10,0.05
--behavior-ratios CSV              default: 0.25,0.30,0.30,0.15
--difficulty-ratios CSV            default: 0.25,0.45,0.30
--strict-quota                     default: true
--show-raw-apps-record             showcase only; default: false
--examples-per-origin INT          showcase only; default: 1
--examples-per-behavior INT        showcase only; default: 1
--resume
--overwrite                       默认 false
--fail-fast
```

三个 ratio 参数的顺序必须在 CLI help 中固定并打印展开后的命名配额；更推荐在 YAML 中使用具名映射，避免裸 CSV 顺序错误。所有 ratio 必须非负、总和为 1。`--strict-quota=true` 表示任一目标单元格不足时整次正式验收失败，而不是用其他数据自动替代。

示例：

```bash
python -m synthesis.cli prepare-apps \
  --apps-path data/raw/apps/train.jsonl \
  --output-dir data/cleaned \
  --split-seed 42

python -m synthesis.cli gate-200 \
  --apps-path data/raw/apps/train.jsonl \
  --output-dir data/gate_200 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --model-revision <PINNED_REVISION> \
  --counterfactual-samples 3 \
  --target-count 200

python -m synthesis.cli export-sft \
  --output-dir data/gate_200 \
  --rationale-mode none

python -m synthesis.cli qa \
  --output-dir data/gate_200 \
  --fail-fast

python -m synthesis.cli showcase \
  --output-dir data/gate_200 \
  --show-raw-apps-record \
  --examples-per-origin 1 \
  --examples-per-behavior 1
```

所有命令必须保存 resolved config、代码版本、模型 revision、随机 seed 和运行环境摘要。`--resume` 以稳定 episode key 幂等恢复；`--overwrite` 不能默认开启。

## 15. 首批 200 条门测

### 15.1 目的

门测不是把完整方案拆成一个简化版本，而是将正式 1200 条配额按比例缩放为 200 条，对完整流水线做小规模验收。单点错误、多点错误、模型自然错误、模型自然正确解和四种行为都必须从第一批开始启用。一条 accepted episode 对应一行 SFT JSONL，因此门测最终恰好导出 200 行；多轮状态切片只用于 QA，不重复计数。

门测来源配额：

| 来源 | 数量 | 比例 |
|---|---:|---:|
| 单点 AST 错误 | 60 | 30% |
| 多点 AST 错误 | 60 | 30% |
| 模型自然错误 | 50 | 25% |
| verified reference 直接提交 | 20 | 10% |
| 模型自然正确解直接提交 | 10 | 5% |
| **合计** | **200** | **100%** |

门测行为配额：

| 行为 | 数量 | 比例 |
|---|---:|---:|
| 提交后直接修复 | 50 | 25% |
| 重放失败样例 | 60 | 30% |
| 提交前主动测试 | 60 | 30% |
| 可靠解直接提交 | 30 | 15% |
| **合计** | **200** | **100%** |

门测完整交叉配额：

| 候选来源 | 提交后直接修复 | 重放失败样例 | 提交前主动测试 | 可靠解直接提交 | 合计 |
|---|---:|---:|---:|---:|---:|
| 单点 AST 错误 | 20 | 20 | 20 | 0 | 60 |
| 多点 AST 错误 | 10 | 25 | 25 | 0 | 60 |
| 模型自然错误 | 20 | 15 | 15 | 0 | 50 |
| verified reference | 0 | 0 | 0 | 20 | 20 |
| 模型自然正确解 | 0 | 0 | 0 | 10 | 10 |
| **合计** | **50** | **60** | **60** | **30** | **200** |

多点错误的 60 条进一步分为：2 个错误 36 条、3 个错误 18 条、4 个错误 6 条。难度分为：`introductory` 50 条、`interview` 90 条、`competition` 60 条。

### 15.2 流程

1. 从 SFT pool 按 difficulty 分层、固定 seed 遍历题目；
2. 清洗并验证 references；
3. 为每题生成单点 AST mutants 和 2～4 点组合 mutants，并执行单点危害性、组合兼容性与遮蔽检查；
4. 让基础模型仅根据公开题面生成自然候选，经 grader 分流为自然错误和自然正确解；
5. 对所有稳定失败候选运行“提交后直接修复”和“失败样例重放”反事实；
6. 对独立的无 grader 反馈状态运行“首次提交前主动设计测试”反事实；
7. 从 verified references 和模型自然正确解构造“已有可靠解时直接提交”；
8. 按交叉配额将通过全部 QA 的轨迹加入对应单元格，直到全部单元格填满；
9. 配额只用于采样，不参与标签判定；不得重标失败样本或插入无效 run；
10. 若某个单元格不足，只可增加候选题、自然生成次数或 proposal 次数继续寻找，且所有样本仍须满足完整条件；
11. 冻结门测输出并生成报告。

同一 problem 默认最多贡献 4 条 accepted trajectories、同一种行为最多 2 条，以避免 200 条被少数题主导。

### 15.3 门测关注指标

门测报告至少包含：

- 原始题数、cleaned 题数、verified reference 比例；
- 单点/多点 mutants 的尝试数、接受数、bug count、遮蔽率及 mutation family 成功率；
- 自然候选生成数、正确率、稳定失败率和不可解析率；
- accepted 200 的行为、候选来源、难度和 I/O 分布；
- ambiguous/discarded 数量与原因；
- without-run/with-run 成功率和 utility 直方图；
- 主动测试 proposal 的解析率、执行率、正效用率；
- 平均每条 accepted 轨迹的模型生成次数、grader 次数和 wall time；
- schema、replay、leakage、mask、dedup 各项错误数；
- supervised token 占比和长度分位数；
- 至少 40 条人工审计结果。若某种行为不足 10 条，则人工审计该行为的全部样本；否则每种行为随机审计至少 10 条。

### 15.4 门测通过条件

全部满足才可扩大生成：

1. 最终恰有 200 条通过自动 QA 的 accepted 轨迹；
2. schema/tool contract 错误为 0；
3. 最终 submit 非 100% 或 replay 失败为 0；
4. 已确认 oracle 泄漏为 0；
5. loss mask 错误为 0，错误历史提交被监督的条数为 0；
6. split 交叉和 APPS test 污染为 0；
7. 单点 exact-edit 和多点 exact-edit-count 验证失败均为 0，多点遮蔽检查失败为 0；
8. “失败样例重放”的 failing-input identity 检查失败为 0；
9. “首次提交前主动测试”的 prompt provenance/白名单检查失败为 0；
10. 四种行为分别达到 50/60/60/30 条，来源分别达到 60/60/50/20/10 条，交叉表每个非零单元格完全满足；
11. 多点错误精确达到 36 条双错误、18 条三错误和 6 条四错误；难度精确达到 50/90/60 条；
12. 反事实 utility 不达阈值却进入训练集的条数为 0；
13. 完全重复 episode 为 0；
14. 人工审计中至少 95% 被判定为“动作由可观察证据支持且轨迹自然”，且无严重问题；
15. 运行成本、失败率和吞吐达到团队预先配置的资源预算。资源预算必须写入 `gate_200.yaml`，不能在运行后倒推。

任何 P0 问题（oracle 泄漏、错误代码参与 loss、最终未通过、split 污染、工具越权）都直接使门测失败。

## 16. 最终验收标准

实现完成需交付以下结果：

### 16.1 功能验收

- 可从 APPS train 原始 JSONL 一条命令生成 cleaned data、互斥 splits、mutants、episodes、SFT JSONL 和 QA report；
- stdin 与 call 两类题均有单元测试和真实样例；
- 两个工具严格遵循固定接口和信息边界；
- 四种行为的判定器有覆盖阈值边界的单元测试；
- 单点、多点和模型自然候选生成器均可在同一次流水线运行中启用；
- 中断后可幂等 resume，不重复计数或覆盖有效 artifact；
- 所有结果可由 pinned config 与 hashes 追溯。

### 16.2 数据验收

- 最终正式数据恰好 1200 条 episode / 1200 行 SFT JSONL；
- 候选来源严格达到 360/360/300/120/60，行为严格达到 300/360/360/180；
- 多点错误严格达到 216 条双错误、108 条三错误和 36 条四错误；
- 难度严格达到 300 条 introductory、540 条 interview 和 360 条 competition；如供给不足则验收失败并提交 shortage report，不得静默改比例；
- 100% 保留轨迹最终 grader accepted 且 replay accepted；
- 100%“失败样例重放”使用最近一次 submit 返回的同一 failing input；
- 100%“首次提交前主动测试”的 input 具有模型 proposal provenance，且 proposal prompt 只含白名单字段；
- 100% 需要反事实判定的行为满足各自阈值；
- 100%“已有可靠解时直接提交”的代码经过双重验证且没有 run；
- 100% 单点 mutant 通过 exact single-edit 验证；
- 100% 多点 mutant 通过 exact edit-count、individual harmfulness、compatibility 和 survivor check；
- 模型自然错误不伪造 mutation label 或 bug count；
- 0 条 ambiguous/discarded 进入 SFT；
- 0 条跨 pool problem 泄漏；
- 0 条 confirmed oracle leakage；
- 0 条错误历史代码作为独立 assistant target。

### 16.3 训练格式验收

- JSONL 和 tokenized 数据均通过 schema；
- 官方 chat template 可无异常渲染全部样本；
- 只有 `trainable=true` 的 assistant span 计算 loss；
- 每个状态切片至少有一个 supervised action；需要执行的样本监督有效 `run_candidate`，所有样本只监督最终正确 submit；
- tool responses 和全部上下文 token 均 mask；
- 随机解码抽检能清楚看到 `state -> tool action -> observation -> submit`，且没有秘密字段。

### 16.4 工程验收

- 自动测试覆盖 loader、adapter、grader projection、tools、mutation、classification、主动验证 prompt whitelist、export 和 loss mask；
- QA 可在不调用生成模型的情况下重放和检查已有 artifacts；
- 日志不会输出 hidden expected/reference 到共享训练日志；
- 报告能解释每一步接受率和主要 reject reason；
- README 给出 200 条门测和正式批量生成的可复制命令。

## 17. 明确不包含的内容

本文档已经把单点错误、多点错误、模型自然错误和自然正确解纳入一次性完整实现。以下内容不属于“数据不完整”，而是会改变实验工具空间或训练方法，因此明确排除：

1. 第三个调试工具、文件系统、Shell、任意 Python scratchpad 或 `inspect_state`；
2. 将 expected output、actual output 或 reference solution放入 `submit` 反馈；
3. 从 hidden tests 中选择输入后伪装成模型自主提出的测试；
4. 把错误代码、失败修复或中间错误提交作为 SFT target；
5. 使用 APPS test split 合成、筛选或调参；
6. 为了填满配额而伪造行为标签或放宽反事实阈值。

可以在不改变数据定义的情况下增加生成规模、增大反事实采样次数、加入更多安全的 AST operator，或从更新后的 policy 持续收集自然错误；这些都应继续遵守同一工具协议、信息边界和验收标准。

## 18. 从 APPS 原始记录到 SFT 数据：完整展示

本节用于让实现者直观看到最终产物。示例分为两部分：

- **真实 APPS 内容**：题目 ID、难度、reference、输入和输出来自可访问的 APPS 数据展示；为控制篇幅，英文题面只保留摘要，实际 loader 必须保留全文。
- **演示性合成内容**：mutation、模型决策、反事实采样次数和 tool trajectory 用于说明输出格式。只有真实脚本运行并通过 grader 后，才能成为正式数据，不能把本文示例直接复制进训练集。

### 18.1 真实 APPS 记录示例

以下使用 APPS 中的 `id=76` 作为展示。该行可在字段等价的 [`4gate/codeparrot_apps` Parquet viewer](https://huggingface.co/datasets/4gate/codeparrot_apps/viewer) 中直接检查；题目来源为 [Codeforces 1369/A](https://codeforces.com/problemset/problem/1369/A)。它的难度为 `interview`，使用标准输入输出，`starter_code` 为空。题意摘要为：给定若干正多边形的边数，判断能否旋转多边形，使至少一条边平行于 x 轴、至少一条边平行于 y 轴。

原始 JSONL 中，`solutions` 和 `input_output` 都是“字符串中的 JSON”，不是已经解析的 Python list/dict。为便于阅读，下面只展示一份 reference：

```json
{
  "id": 76,
  "question": "Lee is going to decorate his house using regular convex polygons... [题面在此处仅作展示性截断]",
  "solutions": "[\"t = int(input())\\nfor _ in range(t):\\n    n = int(input())\\n    if n % 4 == 0:\\n        print('YES')\\n    else:\\n        print('NO')\\n\"]",
  "input_output": "{\"inputs\":[\"4\\n3\\n4\\n12\\n1000000000\\n\"],\"outputs\":[\"NO\\nYES\\nYES\\nYES\\n\"]}",
  "difficulty": "interview",
  "url": "https://codeforces.com/problemset/problem/1369/A",
  "starter_code": ""
}
```

完成两次 `json.loads` 后，内存中的对象应类似：

```python
solutions = [
    "t = int(input())\n"
    "for _ in range(t):\n"
    "    n = int(input())\n"
    "    if n % 4 == 0:\n"
    "        print('YES')\n"
    "    else:\n"
    "        print('NO')\n"
]

input_output = {
    "inputs": ["4\n3\n4\n12\n1000000000\n"],
    "outputs": ["NO\nYES\nYES\nYES\n"],
}
```

这里有一个容易实现错的细节：虽然 stdin 内部有 4 组子问题，但 APPS 的 `inputs` 列表只有 1 个元素。因此 grader 的 `total=1`，不是 4。只要整段 stdout 有一处不匹配，这一个 APPS test item 就失败，`pass_rate=0.0`。

### 18.2 清洗后的题目记录

建议 `cleaned/problems.jsonl` 中把公开字段和私有评测字段明确分开：

```json
{
  "id": "76",
  "public_problem": {
    "question": "<完整英文题面>",
    "starter_code": "",
    "difficulty": "interview",
    "io_mode": "stdin",
    "input_format_note": "Pass one complete stdin string to run_candidate(input)."
  },
  "private_evaluation": {
    "inputs": ["4\n3\n4\n12\n1000000000\n"],
    "outputs": ["NO\nYES\nYES\nYES\n"],
    "verified_reference_indices": [0],
    "selected_reference_index": 0
  },
  "source": {
    "dataset": "codeparrot/apps",
    "split": "train",
    "url": "https://codeforces.com/problemset/problem/1369/A"
  }
}
```

`public_problem` 可以进入模型上下文；`private_evaluation.outputs` 和 reference 索引只允许 grader 与审计程序读取。

### 18.3 单点 AST 错误的具体展示

Reference 中的核心条件为：

```python
if n % 4 == 0:
```

生成器选择一个 AST `Constant` 节点，产生一处语义错误：

```python
t = int(input())
for _ in range(t):
    n = int(input())
    if n % 3 == 0:          # 由 4 改为 3
        print("YES")
    else:
        print("NO")
```

对应内部 edit record：

```json
{
  "candidate_origin": "synthetic_single",
  "bug_count": 1,
  "semantic_edit_count": 1,
  "edits": [
    {
      "operator": "boundary_constant_replace",
      "family": "boundary_constant",
      "node_type": "Constant",
      "before": "4",
      "after": "3",
      "source_span": {"line": 4, "col": 11, "end_line": 4, "end_col": 12}
    }
  ]
}
```

在示例 stdin 上，该 candidate 的实际 stdout 为：

```text
YES
NO
YES
NO
```

grader 内部可以看到：

```json
{
  "passed": 0,
  "total": 1,
  "pass_rate": 0.0,
  "cases": [
    {
      "input": "4\n3\n4\n12\n1000000000\n",
      "expected": "NO\nYES\nYES\nYES\n",
      "actual": "YES\nNO\nYES\nNO\n",
      "passed": false
    }
  ]
}
```

但 `submit` 投影给模型的内容只能是：

```json
{
  "status": "wrong_answer",
  "passed": 0,
  "total": 1,
  "pass_rate": 0.0,
  "failing_input": "4\n3\n4\n12\n1000000000\n"
}
```

注意 public response 中没有 `expected` 和 `actual`。

### 18.4 多点 AST 错误的具体展示

在同一 reference 上组合两个互不重叠的编辑：

1. `range(t)` 改为 `range(t - 1)`，漏掉最后一组输入；
2. `n % 4` 改为 `n % 3`，判断条件错误。

组合 candidate：

```python
t = int(input())
for _ in range(t - 1):      # 错误 1
    n = int(input())
    if n % 3 == 0:          # 错误 2
        print("YES")
    else:
        print("NO")
```

其 edit metadata：

```json
{
  "candidate_origin": "synthetic_multi",
  "bug_count": 2,
  "semantic_edit_count": 2,
  "all_individually_harmful": true,
  "survivor_check_passed": true,
  "masked_mutation": false,
  "edits": [
    {
      "operator": "range_stop_offset",
      "family": "range_bound",
      "before": "range(t)",
      "after": "range(t - 1)"
    },
    {
      "operator": "boundary_constant_replace",
      "family": "boundary_constant",
      "before": "4",
      "after": "3"
    }
  ]
}
```

对同一输入调用 `run_candidate`，模型只能看到自己的程序行为：

```json
{
  "status": "ok",
  "stdout": "YES\nNO\nYES\n",
  "stderr": "",
  "error": null,
  "exit_code": 0,
  "truncated": false
}
```

环境不能在这一响应后追加“缺少最后一行”或“正确输出应该是……”，因为那属于 grader oracle。

### 18.5 模型自然候选的具体展示

自然候选生成器收到的内容只有：

```text
题面全文
starter_code
输入输出模式说明
```

它不接收 reference、tests 或 mutation label。假设模型自然生成：

```python
t = int(input())
for _ in range(t):
    n = int(input())
    print("YES" if n % 2 == 0 else "NO")
```

这个候选没有人工 mutation provenance，因此 metadata 必须是：

```json
{
  "candidate": {
    "origin": "model_natural_failure",
    "generator_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "generator_revision": "<PINNED_REVISION>",
    "generation_seed": 20260017,
    "generation_prompt_hash": "sha256:...",
    "code_hash": "sha256:..."
  },
  "mutation": null,
  "seed_submit": {
    "status": "wrong_answer",
    "passed": 0,
    "total": 1,
    "pass_rate": 0.0,
    "failing_input_hash": "sha256:..."
  }
}
```

不能因为它看起来像“常量 4 写成 2”，就在事后伪造一个 AST mutation record。自然错误和人工错误的来源必须严格区分。

### 18.6 四种训练行为分别长什么样

以下省略 system prompt 和 tool schemas，只突出 messages。反事实次数属于演示格式；正式值必须由模型真实生成和 grader 验证得到。

#### 得到 submit 反馈后直接修复

```text
[USER / 不计算 loss]
Problem:
<完整题面>

Current candidate:
<包含错误的完整代码>

Previous submit result:
pass_rate = 0.0
failing_input = "4\n3\n4\n12\n1000000000\n"

[ASSISTANT / 计算 loss]
submit(code=<将条件恢复为 n % 4 == 0 的完整代码>)

[TOOL / 不计算 loss]
{"status":"accepted","passed":1,"total":1,"pass_rate":1.0,"failing_input":null}
```

这类数据要求无执行分支的真实成功率达到阈值，例如 `3/3`。错误 candidate 只存在于 user context。

#### 重放 submit 返回的失败样例

```text
[USER / 不计算 loss]
Problem: <完整题面>
Current candidate: <失败完整代码>
Previous submit result:
  pass_rate = 0.0
  failing_input = "4\n3\n4\n12\n1000000000\n"

[ASSISTANT / 计算 loss]
run_candidate(input="4\n3\n4\n12\n1000000000\n")

[TOOL / 不计算 loss]
{"status":"ok","stdout":"YES\nNO\nYES\nNO\n","stderr":"","error":null,"exit_code":0,"truncated":false}

[ASSISTANT / 计算 loss]
submit(code=<最终正确完整代码>)

[TOOL / 不计算 loss]
{"status":"accepted","passed":1,"total":1,"pass_rate":1.0,"failing_input":null}
```

此类只有在真实测得类似以下反事实结果后才能保留：

```json
{
  "without_run_successes": 0,
  "with_run_successes": 3,
  "k": 3,
  "without_run_rate": 0.0,
  "with_run_rate": 1.0,
  "utility": 1.0
}
```

这里的数值只是格式示例，不能写死到分类器测试之外的正式数据。

#### 首次 submit 前主动设计测试

模型尚未看到任何 grader 信息。假设它根据题面和 candidate 自主提出最小输入：

```text
[USER / 不计算 loss]
Problem: <完整题面>
Current candidate: <使用 n % 3 == 0 的完整代码>
No grader feedback is available.

[ASSISTANT / 计算 loss]
run_candidate(input="1\n3\n")

[TOOL / 不计算 loss]
{"status":"ok","stdout":"YES\n","stderr":"","error":null,"exit_code":0,"truncated":false}

[ASSISTANT / 计算 loss]
submit(code=<最终正确完整代码>)

[TOOL / 不计算 loss]
{"status":"accepted","passed":1,"total":1,"pass_rate":1.0,"failing_input":null}
```

`"1\n3\n"` 必须能追溯到模型 proposal 原始输出。即使它恰好包含在某个 hidden test 中，也只有在生成器证明模型没有看到 hidden tests 时才可接受。

#### 已有可靠解时直接提交

```text
[USER / 不计算 loss]
Problem: <完整题面>

[ASSISTANT / 计算 loss]
submit(code=<两次验证为 100% 的 reference 或模型自然正确解>)

[TOOL / 不计算 loss]
{"status":"accepted","passed":1,"total":1,"pass_rate":1.0,"failing_input":null}
```

这类数据不包含错误 candidate，也不调用 `run_candidate`。

### 18.7 多错误、多轮 episode 与 loss mask

下面展示一条多错误 episode。第一次修复只解决了 `range(t - 1)`，但仍保留 `% 3`，因此第一次修复提交仍然失败。该错误提交必须保留为后续状态，却不计算 loss。

| 顺序 | role/action | 结果 | `trainable` |
|---:|---|---|---:|
| 0 | system：规则与工具 | context | false |
| 1 | user：题面、双错误 candidate、历史 submit 反馈 | context | false |
| 2 | assistant：`run_candidate(failing_input)` | 获取行为 | true |
| 3 | tool：stdout=`YES/NO/YES` | observation | false |
| 4 | assistant：`submit(code=<只修复 range 的代码>)` | 仍失败 | **false** |
| 5 | tool：新的 pass rate 与 failing input | context | false |
| 6 | assistant：`run_candidate(latest_failing_input)` | 再诊断 | true |
| 7 | tool：stdout=`YES/NO/YES/NO` | observation | false |
| 8 | assistant：`submit(code=<两个错误均修复的代码>)` | accepted | true |
| 9 | tool：accepted | terminal observation | false |

对应消息结构的关键部分：

```json
[
  {
    "role": "user",
    "content": "<problem + current candidate + previous public feedback>",
    "trainable": false
  },
  {
    "role": "assistant",
    "tool_calls": [{"name": "run_candidate", "arguments": {"input": "4\n3\n4\n12\n1000000000\n"}}],
    "trainable": true
  },
  {
    "role": "tool",
    "name": "run_candidate",
    "content": "{\"status\":\"ok\",\"stdout\":\"YES\\nNO\\nYES\\n\",\"stderr\":\"\",\"error\":null,\"exit_code\":0,\"truncated\":false}",
    "trainable": false
  },
  {
    "role": "assistant",
    "tool_calls": [{"name": "submit", "arguments": {"code": "<只修复一个错误的完整代码>"}}],
    "trainable": false
  },
  {
    "role": "tool",
    "name": "submit",
    "content": "{\"status\":\"wrong_answer\",\"passed\":0,\"total\":1,\"pass_rate\":0.0,\"failing_input\":\"4\\n3\\n4\\n12\\n1000000000\\n\"}",
    "trainable": false
  },
  {
    "role": "assistant",
    "tool_calls": [{"name": "run_candidate", "arguments": {"input": "4\n3\n4\n12\n1000000000\n"}}],
    "trainable": true
  },
  {
    "role": "tool",
    "name": "run_candidate",
    "content": "{\"status\":\"ok\",\"stdout\":\"YES\\nNO\\nYES\\nNO\\n\",\"stderr\":\"\",\"error\":null,\"exit_code\":0,\"truncated\":false}",
    "trainable": false
  },
  {
    "role": "assistant",
    "tool_calls": [{"name": "submit", "arguments": {"code": "<最终正确完整代码>"}}],
    "trainable": true
  }
]
```

必须重点检查第 4 个 assistant 回合：它包含错误代码，所以 labels 全为 `-100`。第 2、6、8 个 assistant 回合才是监督目标。第 3、5、7 个 tool 回合虽然不计算 loss，但 `attention_mask=1`，后续 assistant 可以读取并利用这些观察；`trainable=false` 绝不表示从上下文中隐藏该消息。

### 18.8 一条最终 SFT JSONL 的完整外形

为了便于阅读，下面用 `<...>` 省略题面全文和代码全文；真实导出不得省略：

```json
{
  "id": "apps-76-post-submit-failure-replay-7f24c1",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_candidate",
        "description": "Run the current candidate program on one model-provided input and return its observable execution result.",
        "parameters": {
          "type": "object",
          "properties": {"input": {"type": "string"}},
          "required": ["input"],
          "additionalProperties": false
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "submit",
        "description": "Submit a complete solution to the private grader.",
        "parameters": {
          "type": "object",
          "properties": {"code": {"type": "string"}},
          "required": ["code"],
          "additionalProperties": false
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "<agent rules>", "trainable": false},
    {"role": "user", "content": "<full problem + full buggy candidate + public submit feedback>", "trainable": false},
    {"role": "assistant", "tool_calls": [{"name": "run_candidate", "arguments": {"input": "4\n3\n4\n12\n1000000000\n"}}], "trainable": true},
    {"role": "tool", "name": "run_candidate", "content": "{\"status\":\"ok\",\"stdout\":\"YES\\nNO\\nYES\\nNO\\n\",\"stderr\":\"\",\"error\":null,\"exit_code\":0,\"truncated\":false}", "trainable": false},
    {"role": "assistant", "tool_calls": [{"name": "submit", "arguments": {"code": "<full corrected code>"}}], "trainable": true},
    {"role": "tool", "name": "submit", "content": "{\"status\":\"accepted\",\"passed\":1,\"total\":1,\"pass_rate\":1.0,\"failing_input\":null}", "trainable": false}
  ],
  "metadata_ref": "apps-76-post-submit-failure-replay-7f24c1"
}
```

### 18.9 对应 metadata 展示

```json
{
  "schema_version": "1.0",
  "episode_id": "apps-76-post-submit-failure-replay-7f24c1",
  "id": "76",
  "difficulty": "interview",
  "io_mode": "stdin",
  "candidate": {
    "origin": "synthetic_single",
    "code_hash": "sha256:...",
    "is_buggy_context": true
  },
  "mutation": {
    "is_mutated": true,
    "bug_count": 1,
    "semantic_edit_count": 1,
    "edits": [
      {
        "operator": "boundary_constant_replace",
        "family": "boundary_constant",
        "before": "4",
        "after": "3"
      }
    ]
  },
  "counterfactual": {
    "k": 3,
    "without_run_successes": 0,
    "without_run_rate": 0.0,
    "with_run_successes": 3,
    "with_run_rate": 1.0,
    "utility": 1.0,
    "paired_seeds": [11, 22, 33]
  },
  "execution_query": {
    "kind": "submit_failing_case",
    "matches_submit_failing_input": true,
    "proposal_model": null
  },
  "behavior_sequence": [
    "post_submit_failure_replay",
    "post_submit_direct_repair"
  ],
  "final_pass_rate": 1.0,
  "final_replay_passed": true,
  "tool_sequence": ["run_candidate", "submit"],
  "quality": {
    "schema_valid": true,
    "replay_valid": true,
    "leakage_scan_passed": true,
    "loss_mask_valid": true,
    "duplicate_of": null,
    "reject_reasons": []
  }
}
```

这仍是格式示例：正式 counterfactual 数字必须来自真实模型运行，不能因为示例中写了 `0/3 -> 3/3` 就假定该 APPS 题一定属于这一行为。

### 18.10 最终数据集目录应如何展示

完成正式 1200 条合成后，目录应至少包含：

```text
data/final/
├── episodes.jsonl              # 1200 行，完整内部可重放 episode
├── sft_messages.jsonl          # 1200 行，已剥离秘密字段
├── metadata.jsonl              # 1200 行，与 episode_id 一一对应
├── dataset_manifest.json       # 数量与比例总表
├── qa_report.json
├── qa_report.md
└── SYNTHESIS_SHOWCASE.md       # 自动抽取的具体样例展示
```

`SYNTHESIS_SHOWCASE.md` 至少自动抽取：

- 1 条真实 APPS 原始/清洗记录；
- 单点、双点、三点、四点、自然错误各 1 条；
- 四种行为各 1 条；
- stdin 和 call 模式各 1 条（如果 cleaned pool 中存在）；
- 1 条带中间失败提交的多轮 episode；
- 每条样例的消息级 loss mask；
- 对应 metadata 和反事实统计；
- 明确的 `[MODEL VISIBLE]`、`[INTERNAL ONLY]`、`[SUPERVISED]` 标记。

showcase 只能从已经通过 QA 的 artifacts 自动生成，不能使用本文手写示例冒充真实产物。
