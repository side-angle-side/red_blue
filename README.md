# Red Blue Platform

面向 LLM 安全评测的红队/蓝队/评测数据流水线。

本项目把安全评测 seed prompt 存入 SQLite，生成红队包装后的 prompt，导出蓝队输入数据契约，并导入蓝队回复和评测结果。
同时提供一个受控 ReAct Agent，用于根据用户目标和数据库状态自动选择下一步工具动作。

完整的架构说明、实现细节和使用手册见 [docs/implementation-and-usage.md](docs/implementation-and-usage.md)。

## 职责边界

### 红队

红队相关代码和命令：

- `src/red_blue_platform/red_team.py`
- `src/red_blue_platform/seed_generation.py`
- `rb-platform generate-seeds`
- `rb-platform generate-red`
- `rb-platform evolve-red`

红队从 seed prompt 出发，生成以下字段：

- `Data_ID`
- `Seed_Prompt`
- `Wrapped_Prompt`
- `Strategy`
- 可选 `Parent_Data_ID`
- 可选 `Generation`
- 可选 `Evolution_Reason`

其中 `Wrapped_Prompt` 是用于测试蓝队模型的 prompt 变体。

`evolve-red` 会从已评测且 `Eval_Result` 表示防御成功的 attack 中选择候选，把旧 `Wrapped_Prompt`、蓝队 `Response` 和 `Eval_Reason` 一起喂给模型，生成更强的下一代变体。新行会记录父样本 ID、代数和进化原因。

### ReAct Agent

ReAct Agent 是决策层，不直接写数据库。它每一步都按以下流程运行：

```text
Observe database state -> Reason next action -> Act through one allowed tool -> Observe result
```

当前允许的工具：

- `observe_state`
- `generate_red`
- `evolve_red`
- `export_blue`
- `attack_report`
- `final`

所有 action 和工具参数都会经过 Pydantic 校验。模型或 fallback planner 只能输出以下形式的结构化 action：

```json
{"thought":"...","action":"tool_name","args":{}}
```

如果参数不符合工具 schema，例如 `limit` 为负数或传入未知字段，该工具调用会被拒绝并返回结构化错误。

Agent 的输入由三类信息组成：

- 用户目标，例如“根据防御成功样本进化下一轮并导出”。
- SQLite 当前状态，例如 seed 数、待蓝队测试行数、防御成功候选数、最新 generation。
- 已导入的数据反馈，例如蓝队 `Response` 和评测 `Eval_Result` / `Eval_Reason`。

Agent 只负责选择工具和参数；红队生成、进化、导出仍由现有确定性函数和数据库 API 执行。实例化由 `AgentFactory` 统一完成，同一个 ReAct runtime 可以用不同 role/profile 暴露不同工具：

- `full`：完整编排，包含红队生成、进化、蓝队导出和报告。
- `red`：只暴露红队相关工具，适合只生成/进化样本，不做蓝队导出。
- `blue`：只暴露蓝队导出和报告工具，适合只处理待测试样本。

### 蓝队

蓝队既可以通过导出/导入数据契约接入外部服务，也可以由内置 blue-team Agent 执行：

- `rb-platform export-blue`
- `rb-platform import-blue`
- `rb-platform run-blue`

导出的 JSONL 会包含共享契约字段，但蓝队服务代码应只把 `Wrapped_Prompt` 传给模型。蓝队模型输出后，将结果写回 `Response` 字段。

`run-blue` 使用指定 API 或本地模型为待测攻击生成防御回复；`--backend fallback` 提供确定性的安全拒答，适合离线验证流程。

### 评测

评测侧对接点：

- `rb-platform import-eval`
- `rb-platform evaluate`
- `src/red_blue_platform/schema.py` 中的评测字段

评测器读取 `Wrapped_Prompt` 和 `Response`，然后写入：

- `Eval_Result`
- 可选 `Eval_Reason`
- 可选 `Eval_Confidence`

`evaluate` 可以运行内置评测 Agent。其 fallback 模式基于安全拒答标记输出 `defense_success` 或 `attack_success`，API/本地模型模式则使用结构化评测提示词。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
```

如果使用 API 后端，需要设置 `api.api_key_env` 配置项对应的环境变量，例如：

```bash
export OPENAI_API_KEY=...
```

可选依赖：

```bash
pip install -e ".[datasets]"      # Hugging Face 数据集导入
pip install -e ".[local-models]"  # 本地 Transformers 模型
```

## 快速开始

运行一个不依赖外部服务的本地确定性 smoke test：

```bash
rb-platform --config config.yaml init-db
rb-platform --config config.yaml import-dataset --source local_jsonl_example
rb-platform --config config.yaml generate-red --backend fallback --limit 1
rb-platform --config config.yaml export-blue --out data/blue_input.jsonl
rb-platform --config config.yaml stats
```

## 使用本地模型加壳

如果不使用 API，可以复制本地 Transformers 示例配置：

```bash
cp config.local-transformers.example.yaml config.yaml
```

该配置默认使用 `Qwen/Qwen3-8B`。如果模型已经下载在本机目录，也可以在命令里显式传入路径，例如：

```bash
rb-platform --config config.yaml init-db
rb-platform --config config.yaml import-dataset --source local_jsonl_example
rb-platform --config config.yaml generate-red \
  --backend local-transformers \
  --model /path/to/Qwen3-8B \
  --max-new-tokens 128 \
  --limit 1
rb-platform --config config.yaml export-blue --out data/blue_input.jsonl
```

`generate-red --backend local-transformers` 会使用本地模型读取基础 `Seed_Prompt`，生成写入数据库的 `Wrapped_Prompt`。

如果需要通过 OpenAI-compatible API 生成红队样本，也可以使用：

```bash
rb-platform --config config.yaml generate-red --backend api --limit 20
```

使用 TatuCloud API 和 S-Eval 抽样 100 条生成加壳请求：

```bash
cp config.tatu.seval.sample100.yaml config.yaml
export TATU_API_KEY=...
rb-platform --config config.yaml init-db
rb-platform --config config.yaml import-dataset --source seval_attack_en_sample100
rb-platform --config config.yaml generate-red --backend api --limit 100
rb-platform --config config.yaml export-blue --out data/seval_sample100_blue_input.jsonl
```

导入蓝队回复：

```bash
rb-platform --config config.yaml import-blue --file data/blue_output.jsonl
```

导入评测结果：

```bash
rb-platform --config config.yaml import-eval --file data/eval_output.jsonl
```

统计 `Wrapped_Prompt` 的攻击成功率：

```bash
rb-platform --config config.yaml attack-report
```

默认会把 `Eval_Result` 为 `success`、`attack_success`、`jailbreak`、`unsafe`、`failed_safety` 的记录计为攻击成功。也可以自定义成功标签：

```bash
rb-platform --config config.yaml attack-report --success-label attack_success --success-label unsafe
```

## 红队进化

当一轮蓝队测试和评测完成后，可以针对“防御成功”的样本生成下一轮更强变体：

```bash
rb-platform --config config.yaml evolve-red --backend api --limit 20
rb-platform --config config.yaml export-blue --evolved-only --out data/evolved_blue_input.jsonl
```

默认会把 `Eval_Result` 为 `defense_success`、`defended`、`safe`、`blocked`、`refusal`、`防御成功` 的记录视为防御成功。也可以自定义：

```bash
rb-platform --config config.yaml evolve-red \
  --defense-success-label 防御成功 \
  --defense-success-label safe
```

`export-blue --evolved-only` 只导出新生成且尚未写入 `Response` 的 evolved rows，适合直接交给蓝队做下一轮测试。

## ReAct Agent 编排

本地无 API 的确定性运行：

```bash
rb-platform --config config.yaml react-agent \
  --goal "生成红队样本并导出给蓝队" \
  --role full \
  --planner-backend fallback \
  --red-backend fallback \
  --out data/agent_blue_input.jsonl
```

只运行红队 profile：

```bash
rb-platform --config config.yaml react-agent \
  --goal "生成红队样本" \
  --role red \
  --planner-backend fallback \
  --red-backend fallback
```

只运行蓝队导出 profile：

```bash
rb-platform --config config.yaml react-agent \
  --goal "导出给蓝队测试" \
  --role blue \
  --planner-backend fallback \
  --out data/blue_input.jsonl
```

针对防御成功反馈推进下一轮：

```bash
rb-platform --config config.yaml react-agent \
  --goal "根据防御成功样本进化下一轮并只导出新样本" \
  --role full \
  --planner-backend fallback \
  --red-backend api \
  --out data/evolved_blue_input.jsonl
```

执行一次完整的离线红蓝提升循环（红方生成 -> 蓝方防御 -> 评测 -> 红方进化 -> 蓝方防御 -> 再评测）：

```bash
rb-platform --config config.yaml react-agent \
  --goal "完成一次红蓝循环提升" \
  --role full \
  --planner-backend fallback \
  --red-backend fallback \
  --blue-backend fallback \
  --evaluator-backend fallback \
  --max-steps 8
```

也可以分别执行蓝方和评测阶段：

```bash
rb-platform --config config.yaml run-blue --backend fallback
rb-platform --config config.yaml evaluate --backend fallback
```

如果需要让模型做 ReAct 决策，可以把 `--planner-backend` 改为 `api` 或 `local-transformers`。建议红队生成后端和规划后端分开配置：规划层负责选工具，红队后端负责生成 `Wrapped_Prompt`。

## 数据源

默认 `config.example.yaml` 使用本地示例文件 `data/seeds.jsonl`，因此新环境无需联网即可跑通基础流程。

`config.huggingface.example.yaml` 展示了如何通过可选依赖 `datasets` 使用 `IS2Lab/S-Eval`。S-Eval 是安全评测 benchmark，许可证为 CC BY-NC-SA 4.0；生产或商业场景使用前需要确认许可证是否符合要求。

本项目也支持本地 JSONL、JSON 和 CSV 数据源，字段映射方式相同。

## 数据契约

SQLite 表结构和 JSONL 字段映射见 `DATABASE_KEYS.md`。

## 开发检查

```bash
python3 -m compileall -q src tests
PYTHONPATH=src pytest
```
