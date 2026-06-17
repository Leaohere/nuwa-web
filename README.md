# 女娲 · Nuwa Web

> 🧬 蒸馏任何人的思维方式 —— 输入人名，自动多 Agent 并行研究，提取心智模型、决策启发式、表达DNA

基于 [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) (22K+ Star) 的理念，构建为可直接使用的网页应用。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 Key:
#   ANTHROPIC_API_KEY=sk-ant-...  (https://console.anthropic.com/)
#   TAVILY_API_KEY=tvly-...       (https://app.tavily.com/ 免费 1000次/月)
```

### 3. 启动

```bash
python app.py
```

### 4. 使用

浏览器打开 http://localhost:8000，输入公众人物名字，点击「开始蒸馏」。

## 流水线架构

```
用户输入人名
    ↓
Phase 1: 六路并行研究 (Agent Swarm)
  ├─ 📚 著作采集 → 书籍/论文/长文
  ├─ 🎙️ 对话采集 → 播客/访谈/AMA
  ├─ 🧬 表达DNA  → Twitter/微博/短文
  ├─ 👁️ 他者视角 → 批评/深度分析
  ├─ ⚖️ 决策记录 → 重大决策/争议
  └─ 📅 时间线   → 生平关键节点
    ↓
Phase 2: 三重验证提炼
  ├─ 跨域复现 (≥2 个领域出现)
  ├─ 生成力 (能预测立场)
  └─ 排他性 (非普适智慧)
    ↓
Phase 3: 生成五层认知画像
  ├─ L1: 表达DNA
  ├─ L2: 心智模型 (3-7)
  ├─ L3: 决策启发式 (5-10)
  ├─ L4: 反模式/价值观底线
  └─ L5: 诚实边界
    ↓
Phase 4: 质量验证
```

## 技术栈

- **后端**: Python FastAPI + Uvicorn
- **前端**: 原生 HTML/CSS/JS (零构建)
- **LLM**: Anthropic Claude API
- **搜索**: Tavily Search API
- **实时进度**: Server-Sent Events (SSE)

## 项目结构

```
nuwa-web/
├── app.py           # FastAPI 主应用 + 前端页面
├── agents.py        # 6 个研究 Agent 定义与执行
├── pipeline.py      # 蒸馏流水线 (提炼 + 验证)
├── config.py        # 配置管理
├── requirements.txt # 依赖清单
└── .env.example     # API Key 配置模板
```

## 诚实边界

- 蒸馏不了直觉——框架能提取，灵感不能
- 捕捉不了突变——截止到调研时间的快照
- 公开表达 ≠ 真实想法——只能基于公开信息
