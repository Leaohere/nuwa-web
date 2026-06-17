"""
女娲 Nuwa Web — 蒸馏任何人的思维方式
FastAPI 主应用 + 内嵌前端页面

启动: python app.py
访问: http://localhost:8000
"""

import asyncio
import json
import sys
import io

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import uuid
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from openai import AsyncOpenAI

import config
from agents import run_all_agents
from pipeline import (
    synthesize_cognitive_profile,
    run_quality_check,
    profile_to_dict,
    TimelineEvent,
)

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(title="女娲 Nuwa", description="蒸馏任何人的思维方式", version="1.0.0")

# 任务存储 (内存)
tasks: dict[str, dict] = {}

# 验证 API Keys
def check_config():
    missing = []
    if not config.LLM_API_KEY or "xxxxxxxx" in config.LLM_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not config.TAVILY_API_KEY or "xxxxxxxx" in config.TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing

# ============================================================
# 人名筛查 (Phase 0)
# ============================================================

# 明显不是人名的黑名单
NON_PERSON_PATTERNS = [
    "是什么", "为什么", "怎么办", "怎么", "怎样", "如何",
    "what is", "how to", "why is",
    "?", "？", "推荐", "教程", "方法", "技巧",
]

async def validate_person_name(name: str) -> dict:
    """
    验证输入是否为真实人物名称，并检测重名。
    返回:
      {"is_person": true/false, "ambiguous": true/false, "options": [...], "reason": ""}
    """
    # 1. 快速规则检查
    if len(name) < 2:
        return {"is_person": False, "ambiguous": False, "options": [], "reason": f"「{name}」太短，不像真实人名。请输入完整姓名。"}
    if len(name) > 60:
        return {"is_person": False, "ambiguous": False, "options": [], "reason": f"「{name}」太长，不像人名。请输入姓名。"}
    if any(p in name for p in NON_PERSON_PATTERNS):
        return {"is_person": False, "ambiguous": False, "options": [], "reason": f"「{name}」看起来是一个问题而不是人名。请输入公众人物姓名。"}
    if name.isdigit():
        return {"is_person": False, "ambiguous": False, "options": [], "reason": f"「{name}」是数字，不是人名。"}

    # 2. LLM 验证 + 重名检测
    missing = check_config()
    if missing:
        return {"is_person": True, "ambiguous": False, "options": [], "reason": ""}

    try:
        client = AsyncOpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        resp = await client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "system",
                "content": """你是一个人名校验器。判断输入是否为公众人物姓名，并检测是否存在重名。

请只回复 JSON:
{
  "is_person": true/false,
  "ambiguous": true/false,
  "corrected_name": "标准姓名",
  "options": [
    {"name": "全名", "identity": "身份描述（行业+职位/角色）", "known_for": "最知名的事迹/作品"},
  ],
  "reason": "非人物时给出原因"
}

规则：
- 如果输入明显不是人名（问题/抽象概念/物品），is_person=false
- 如果存在多个知名同名人物，ambiguous=true，穷举所有你能确认的选项（至少列出所有主要领域的）
- 覆盖所有领域：商界、政界、学术界、体育界、艺术界、娱乐圈、科技界、文学界等
- 如果只有一个知名人物，ambiguous=false
- 虚构人物也算人物（如孙悟空），但要标注
- option 的 identity 要具体：不要只写"演员"，写"中国男演员，代表作《琅琊榜》"
- 如果有多个同名同领域的人物，都要列出"""
            }, {
                "role": "user",
                "content": f"请校验：「{name}」"
            }]
        )
        text = resp.choices[0].message.content.strip()

        # 解析 JSON
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in text:
                si = text.find(start_char)
                ei = text.rfind(end_char) + 1
                try:
                    return json.loads(text[si:ei])
                except json.JSONDecodeError:
                    continue
        return json.loads(text)

    except Exception:
        return {"is_person": True, "ambiguous": False, "options": [], "reason": ""}


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查 + 配置状态"""
    missing = check_config()
    return {
        "status": "ok" if not missing else "missing_config",
        "missing_keys": missing,
        "model": config.LLM_MODEL,
        "message": "女娲已就绪，可以开始蒸馏 🧬" if not missing else f"请先配置: {', '.join(missing)}"
    }


@app.post("/api/research")
async def start_research(request: Request):
    """启动研究任务"""
    missing = check_config()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"缺少 API Key 配置: {', '.join(missing)}。请复制 .env.example 为 .env 并填入你的 Key。"
        )

    body = await request.json()
    target_name = body.get("name", "").strip()
    confirmed_identity = body.get("confirmed", "").strip()  # 用户选定的身份
    if not target_name:
        raise HTTPException(status_code=400, detail="请输入人名")

    # Phase 0: 人名筛查 + 重名检测
    if not confirmed_identity:
        validation = await validate_person_name(target_name)

        if not validation.get("is_person", True):
            raise HTTPException(status_code=400, detail=validation.get("reason", "不是有效人名"))

        if validation.get("ambiguous", False):
            options = validation.get("options", [])
            if options:
                return {
                    "ambiguous": True,
                    "message": f"「{target_name}」存在多位知名人物，请指定你要蒸馏的是谁：",
                    "options": options,
                }

        # 有修正名则使用
        corrected = validation.get("corrected_name", "")
        if corrected:
            target_name = corrected

    else:
        # 用户已指定身份，拼接搜索名
        target_name = f"{target_name} {confirmed_identity}"

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "id": task_id,
        "name": target_name,
        "status": "starting",
        "progress": [],
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }

    # 后台启动流水线
    asyncio.create_task(run_pipeline(task_id, target_name))

    return {"task_id": task_id, "status": "started"}


@app.get("/api/research/{task_id}/stream")
async def stream_research(task_id: str):
    """SSE 实时进度流"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]

    async def event_generator():
        # 发送已有的进度事件
        for evt in task["progress"]:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        # 持续监听新事件
        last_idx = len(task["progress"])
        while task["status"] not in ("completed", "failed"):
            await asyncio.sleep(0.3)
            while last_idx < len(task["progress"]):
                evt = task["progress"][last_idx]
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                last_idx += 1

        # 发送最终事件
        if task["status"] == "completed":
            yield f"data: {json.dumps({'type': 'complete', 'task_id': task_id}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': task.get('error', '未知错误')}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/research/{task_id}/result")
async def get_result(task_id: str):
    """获取最终研究结果"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task["status"] == "completed" and task["result"]:
        return task["result"]
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task.get("error", "任务失败"))
    else:
        return {"status": task["status"], "message": "任务仍在进行中"}


# ============================================================
# 流水线执行 (后台任务)
# ============================================================

async def run_pipeline(task_id: str, target_name: str):
    """后台执行完整的五阶段蒸馏流水线"""
    task = tasks[task_id]
    progress_queue = asyncio.Queue()

    try:
        # 初始化 LLM 客户端 (OpenAI 兼容接口)
        client = AsyncOpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

        # Phase 0: 通知开始
        task["status"] = "researching"
        await push_progress(task, {
            "type": "phase",
            "phase": 0,
            "label": f"开始蒸馏「{target_name}」",
            "status": "running"
        })

        # Phase 1: 六路并行研究
        await push_progress(task, {
            "type": "phase",
            "phase": 1,
            "label": "六路并行研究",
            "status": "running"
        })

        # 启动进度转发协程
        async def forward_progress():
            while True:
                evt = await progress_queue.get()
                await push_progress(task, evt)
                progress_queue.task_done()

        forwarder = asyncio.create_task(forward_progress())

        agent_results = await run_all_agents(
            target_name=target_name,
            progress_queue=progress_queue,
            llm_client=client,
        )

        await push_progress(task, {
            "type": "phase",
            "phase": 1,
            "label": "六路并行研究",
            "status": "completed",
            "agents_done": sum(1 for r in agent_results.values() if r.error is None),
            "agents_error": sum(1 for r in agent_results.values() if r.error is not None),
        })

        # Phase 2-3: 提炼 + 生成认知画像
        task["status"] = "synthesizing"
        profile = await synthesize_cognitive_profile(
            target_name=target_name,
            agent_results=agent_results,
            llm_client=client,
            progress_queue=progress_queue,
        )
        profile.generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat()

        # 从原始数据中提取时间线
        timeline_data = agent_results.get("06-timeline")
        if timeline_data and isinstance(timeline_data.key_findings, dict):
            tld = timeline_data.key_findings
            for m in tld.get("milestones", [])[:15]:
                profile.timeline.append(TimelineEvent(
                    year=str(m.get("year", "")),
                    event=str(m.get("event", "")),
                    significance=str(m.get("significance", "")),
                ))

        # Phase 4: 质量验证
        task["status"] = "validating"
        quality = await run_quality_check(profile, client, progress_queue)

        # 组装最终结果
        result = profile_to_dict(profile)
        result["quality"] = quality
        result["task_id"] = task_id

        task["result"] = result
        task["status"] = "completed"

        # 关闭转发器
        forwarder.cancel()
        try:
            await forwarder
        except asyncio.CancelledError:
            pass

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        await push_progress(task, {
            "type": "error",
            "message": str(e)
        })


async def push_progress(task: dict, event: dict):
    """推送进度事件到任务"""
    event["_ts"] = time.time()
    task["progress"].append(event)


# ============================================================
# 前端页面
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>女娲 · Nuwa — 蒸馏任何人的思维方式</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #131320;
    --surface2: #1a1a2e;
    --border: #2a2a40;
    --text: #e0e0f0;
    --text2: #9898b8;
    --accent: #a78bfa;
    --accent2: #7c3aed;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #f87171;
    --blue: #60a5fa;
    --pink: #f472b6;
    --orange: #fb923c;
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
  }

  /* Background glow */
  body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
    position: relative;
    z-index: 1;
  }

  /* Header */
  .header {
    text-align: center;
    margin-bottom: 36px;
  }
  .header .logo {
    font-size: 48px;
    margin-bottom: 8px;
  }
  .header h1 {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd, #a78bfa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .header .subtitle {
    color: var(--text2);
    font-size: 16px;
    margin-top: 4px;
  }
  .header .tagline {
    color: var(--text2);
    font-size: 14px;
    margin-top: 8px;
    font-style: italic;
    opacity: 0.7;
  }

  /* Input area */
  .input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 32px;
  }
  .input-row {
    display: flex;
    gap: 12px;
    align-items: center;
  }
  .input-row input {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    color: var(--text);
    font-size: 16px;
    outline: none;
    transition: border-color 0.2s;
  }
  .input-row input:focus {
    border-color: var(--accent);
  }
  .input-row input::placeholder {
    color: #555;
  }
  .btn-distill {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    border: none;
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 14px 28px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
  }
  .btn-distill:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
  }
  .btn-distill:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
  .config-warning {
    margin-top: 12px;
    padding: 10px 14px;
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: var(--radius-sm);
    color: var(--yellow);
    font-size: 13px;
    display: none;
  }

  /* Progress area */
  .progress-area {
    display: none;
    margin-bottom: 32px;
  }
  .progress-area.active {
    display: block;
  }
  .phase-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .phase-label .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .agent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
  }
  .agent-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px;
    transition: all 0.3s;
  }
  .agent-card.running {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.15);
  }
  .agent-card.done {
    border-color: var(--green);
    opacity: 0.9;
  }
  .agent-card.error {
    border-color: var(--red);
    opacity: 0.8;
  }
  .agent-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .agent-icon { font-size: 20px; }
  .agent-name {
    font-weight: 600;
    font-size: 14px;
  }
  .agent-status {
    font-size: 12px;
    color: var(--text2);
    margin-left: auto;
  }
  .agent-card.running .agent-status { color: var(--accent); }
  .agent-card.done .agent-status { color: var(--green); }
  .agent-card.error .agent-status { color: var(--red); }
  .agent-sources {
    font-size: 12px;
    color: var(--text2);
    margin-top: 4px;
  }

  /* Results area */
  .results-area {
    display: none;
  }
  .results-area.active {
    display: block;
  }

  /* Section cards */
  .section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
  }
  .section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title .icon { font-size: 22px; }

  /* One-liner */
  .one-liner {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(167,139,250,0.05));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 24px;
    font-size: 18px;
    font-style: italic;
    text-align: center;
    color: #c4b5fd;
  }

  /* Expression DNA */
  .dna-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
  }
  .dna-item {
    background: var(--surface2);
    border-radius: var(--radius-sm);
    padding: 14px;
  }
  .dna-label {
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text2);
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }
  .dna-value {
    font-size: 14px;
    color: var(--text);
  }
  .signature-phrases {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }
  .phrase-tag {
    background: rgba(167,139,250,0.15);
    color: #c4b5fd;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 13px;
  }

  /* Mental Models */
  .model-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 16px;
  }
  .model-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    transition: border-color 0.2s;
  }
  .model-card:hover {
    border-color: var(--accent);
  }
  .model-name {
    font-size: 17px;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 8px;
  }
  .model-desc {
    font-size: 14px;
    color: var(--text);
    margin-bottom: 12px;
    line-height: 1.5;
  }
  .model-details {
    font-size: 13px;
    color: var(--text2);
  }
  .model-details ul {
    list-style: none;
    padding: 0;
  }
  .model-details li {
    padding: 3px 0;
  }
  .model-details li::before {
    content: "▸ ";
    color: var(--accent);
  }
  .model-limitations {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .model-limitations .limit-title {
    font-size: 12px;
    color: var(--yellow);
    margin-bottom: 4px;
  }

  /* Heuristics */
  .heuristic-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .heuristic-item {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
  }
  .heuristic-rule {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .heuristic-context {
    font-size: 12px;
    color: var(--text2);
  }

  /* Timeline */
  .timeline-visual {
    position: relative;
    padding-left: 24px;
  }
  .timeline-visual::before {
    content: '';
    position: absolute;
    left: 8px; top: 0; bottom: 0;
    width: 2px;
    background: var(--border);
  }
  .timeline-item {
    position: relative;
    padding: 8px 0 8px 24px;
    font-size: 14px;
  }
  .timeline-item::before {
    content: '';
    position: absolute;
    left: -20px; top: 14px;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
  }
  .timeline-year {
    font-weight: 700;
    color: var(--accent);
    font-size: 13px;
  }
  .timeline-event {
    color: var(--text);
  }
  .timeline-sig {
    color: var(--text2);
    font-size: 12px;
  }

  /* Anti-patterns & Boundaries */
  .list-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .list-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 14px;
    padding: 8px 0;
  }
  .list-item .bullet {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
  }
  .bullet.red { background: var(--red); }
  .bullet.yellow { background: var(--yellow); }
  .bullet.blue { background: var(--blue); }

  /* Tensions */
  .tension-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--surface2);
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    font-size: 14px;
  }

  /* Quality badge */
  .quality-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }
  .quality-badge.pass { background: rgba(52,211,153,0.15); color: var(--green); }
  .quality-badge.needs_revision { background: rgba(251,191,36,0.15); color: var(--yellow); }
  .quality-badge.fail { background: rgba(248,113,113,0.15); color: var(--red); }

  /* Source summary */
  .source-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
  }
  .source-stat {
    background: var(--surface2);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    text-align: center;
  }
  .source-stat .count {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
  }
  .source-stat .label {
    font-size: 11px;
    color: var(--text2);
    margin-top: 2px;
  }

  /* Error */
  .error-banner {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: var(--radius-sm);
    padding: 16px;
    color: var(--red);
    font-size: 14px;
    display: none;
    margin-bottom: 20px;
  }

  /* Responsive */
  @media (max-width: 640px) {
    .container { padding: 20px 12px; }
    .header h1 { font-size: 28px; }
    .input-row { flex-direction: column; }
    .btn-distill { width: 100%; text-align: center; }
    .model-cards { grid-template-columns: 1fr; }
    .agent-grid { grid-template-columns: 1fr; }
  }

  footer {
    text-align: center;
    color: var(--text2);
    font-size: 12px;
    margin-top: 48px;
    padding: 24px 0;
    opacity: 0.6;
  }
  footer a { color: var(--accent); }
</style>
</head>
<body>

<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="logo">🧬</div>
    <h1>女娲 · Nuwa</h1>
    <p class="subtitle">蒸馏任何人的思维方式</p>
    <p class="tagline">输入人名 → 六路并行研究 → 提取心智模型 · 决策启发式 · 表达DNA</p>
  </div>

  <!-- Input -->
  <div class="input-card">
    <div class="input-row">
      <input type="text" id="nameInput" placeholder="输入一个公众人物名字，如：张一鸣、乔布斯、费曼..." />
      <button class="btn-distill" id="distillBtn" onclick="startDistill()">
        ⚡ 开始蒸馏
      </button>
    </div>
    <div class="config-warning" id="configWarning">
      ⚠️ 请先配置 .env 文件中的 ANTHROPIC_API_KEY 和 TAVILY_API_KEY
    </div>
    <div class="error-banner" id="errorBanner"></div>
  </div>

  <!-- Progress -->
  <div class="progress-area" id="progressArea">
    <div class="phase-label" id="phaseLabel">
      <div class="spinner"></div>
      <span id="phaseText">准备中...</span>
    </div>
    <div class="agent-grid" id="agentGrid"></div>
  </div>

  <!-- Results -->
  <div class="results-area" id="resultsArea"></div>

  <footer>
    基于 <a href="https://github.com/alchaincyf/nuwa-skill" target="_blank">nuwa-skill</a> 理念构建 ·
    仅使用公开信息 · 蒸馏不了直觉，捕捉不了突变 · 诚实比酷更重要
  </footer>
</div>

<script>
// ============================================================
// 状态管理
// ============================================================
let currentTaskId = null;
let eventSource = null;

const AGENT_META = {
  '01-writings':      { name: '著作采集', icon: '📚' },
  '02-conversations': { name: '对话采集', icon: '🎙️' },
  '03-expression-dna':{ name: '表达DNA', icon: '🧬' },
  '04-external-views':{ name: '他者视角', icon: '👁️' },
  '05-decisions':     { name: '决策记录', icon: '⚖️' },
  '06-timeline':      { name: '时间线', icon: '📅' }
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  document.getElementById('nameInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') startDistill();
  });
});

async function checkHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    if (data.missing_keys && data.missing_keys.length > 0) {
      document.getElementById('configWarning').style.display = 'block';
      document.getElementById('configWarning').textContent =
        `⚠️ 缺少配置: ${data.missing_keys.join(', ')}。请复制 .env.example 为 .env 并填入你的 Key`;
      document.getElementById('distillBtn').disabled = true;
    }
  } catch(e) {
    console.error('Health check failed', e);
  }
}

// ============================================================
// 开始蒸馏
// ============================================================
// 快速本地筛查关键词
const NON_PERSON_WORDS = ['是什么','为什么','怎么办','怎么','怎样','如何','教程','方法','技巧','推荐','what','how','why'];

async function startDistill() {
  const nameInput = document.getElementById('nameInput');
  const name = nameInput.value.trim();
  if (!name) return;

  // 快速本地筛查
  if (name.length < 2) {
    showError('「' + name + '」太短，不像真实人名。请输入完整姓名。');
    return;
  }
  if (name.length > 60) {
    showError('输入太长，不像人名。请输入姓名。');
    return;
  }
  const lower = name.toLowerCase();
  for (const w of NON_PERSON_WORDS) {
    if (lower.includes(w)) {
      showError('「' + name + '」看起来是一个问题/关键词，不是人名。请输入公众人物姓名。');
      return;
    }
  }

  // 重置 UI
  document.getElementById('errorBanner').style.display = 'none';
  document.getElementById('resultsArea').classList.remove('active');
  document.getElementById('resultsArea').innerHTML = '';
  document.getElementById('progressArea').classList.add('active');
  document.getElementById('phaseText').textContent = '启动中...';
  document.getElementById('distillBtn').disabled = true;
  document.getElementById('distillBtn').textContent = '⏳ 蒸馏中...';

  // 初始化 Agent 卡片
  const grid = document.getElementById('agentGrid');
  grid.innerHTML = '';
  for (const [id, meta] of Object.entries(AGENT_META)) {
    grid.innerHTML += `
      <div class="agent-card" id="agent-${id}">
        <div class="agent-header">
          <span class="agent-icon">${meta.icon}</span>
          <span class="agent-name">${meta.name}</span>
          <span class="agent-status">等待中</span>
        </div>
        <div class="agent-sources"></div>
      </div>`;
  }

  try {
    const resp = await fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, confirmed: currentConfirmed || '' })
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || '启动失败');
    }

    const data = await resp.json();

    // 重名检测：显示选项让用户选择
    if (data.ambiguous && data.options) {
      showDisambiguation(name, data.options);
      resetButton();
      return;
    }

    const { task_id } = data;
    currentTaskId = task_id;
    currentConfirmed = '';

    // 连接 SSE
    connectSSE(task_id);
  } catch(e) {
    showError(e.message);
    resetButton();
  }
}

let currentConfirmed = '';

function showDisambiguation(name, options) {
  document.getElementById('progressArea').classList.add('active');
  document.getElementById('phaseText').textContent = '「' + name + '」存在 ' + options.length + ' 位知名人物，请选择你要蒸馏的是谁：';
  document.querySelector('.spinner').style.display = 'none';

  // 按领域分类
  const categories = {
    '商界': [],
    '科技界': [],
    '政界': [],
    '学术界': [],
    '娱乐圈': [],
    '体育界': [],
    '艺术界': [],
    '文学界': [],
    '其他': []
  };
  for (const opt of options) {
    const id = (opt.identity || '').toLowerCase();
    if (/创|商|企|CEO|投资|资本|总裁|董事|老板/.test(id)) categories['商界'].push(opt);
    else if (/科技|程序|工程|CTO|技术|开发|互联网|AI|软件/.test(id)) categories['科技界'].push(opt);
    else if (/政|主席|书记|总理|部长|总统|市长|委员/.test(id)) categories['政界'].push(opt);
    else if (/教授|学者|博士|院士|研究|科学/.test(id)) categories['学术界'].push(opt);
    else if (/演员|歌手|导演|明星|艺人|主持|影视|综艺/.test(id)) categories['娱乐圈'].push(opt);
    else if (/球员|运动员|教练|体育|足球|篮球|游泳|田径/.test(id)) categories['体育界'].push(opt);
    else if (/画家|艺术|设计|音乐|摄影|雕塑/.test(id)) categories['艺术界'].push(opt);
    else if (/作家|诗人|作者|小说|文学/.test(id)) categories['文学界'].push(opt);
    else categories['其他'].push(opt);
  }

  let html = '<div style="max-height:500px;overflow-y:auto;padding-right:8px;">';
  for (const [cat, items] of Object.entries(categories)) {
    if (!items.length) continue;
    html += `<div style="margin-bottom:16px;">
      <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;padding-left:4px;">${cat} (${items.length})</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;">`;
    for (const opt of items) {
      html += `
        <div class="agent-card" style="cursor:pointer;border-color:var(--accent);transition:all 0.15s;"
             onclick="selectIdentity('${e(name)}', '${e(opt.identity)}')"
             onmouseenter="this.style.borderColor='#c4b5fd';this.style.background='var(--surface2)'"
             onmouseleave="this.style.borderColor='var(--accent)';this.style.background='var(--surface)'">
          <div style="font-weight:600;font-size:14px;">${e(opt.name)}</div>
          <div style="font-size:12px;color:var(--accent);margin-top:3px;">${e(opt.identity)}</div>
          ${opt.known_for ? `<div style="font-size:11px;color:var(--text2);margin-top:2px;">${e(opt.known_for)}</div>` : ''}
        </div>`;
    }
    html += '</div></div>';
  }
  html += '</div>';

  document.getElementById('agentGrid').innerHTML = html;
  // 改用更宽的展示区
  document.getElementById('agentGrid').style.gridTemplateColumns = '1fr';
}

function selectIdentity(name, identity) {
  currentConfirmed = identity;
  document.getElementById('progressArea').classList.remove('active');
  document.getElementById('distillBtn').disabled = true;
  document.getElementById('distillBtn').textContent = '⏳ 蒸馏中...';
  document.getElementById('nameInput').value = name + ' — ' + identity;
  startDistillWithConfirm(name, identity);
}

function startDistillWithConfirm(name, identity) {
  // 重置 UI
  document.getElementById('errorBanner').style.display = 'none';
  document.getElementById('resultsArea').classList.remove('active');
  document.getElementById('resultsArea').innerHTML = '';
  document.getElementById('progressArea').classList.add('active');
  document.getElementById('phaseText').textContent = '启动中...';
  document.querySelector('.spinner').style.display = 'inline-block';

  const grid = document.getElementById('agentGrid');
  grid.innerHTML = '';
  for (const [id, meta] of Object.entries(AGENT_META)) {
    grid.innerHTML += `
      <div class="agent-card" id="agent-${id}">
        <div class="agent-header">
          <span class="agent-icon">${meta.icon}</span>
          <span class="agent-name">${meta.name}</span>
          <span class="agent-status">等待中</span>
        </div>
        <div class="agent-sources"></div>
      </div>`;
  }

  fetch('/api/research', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name, confirmed: identity })
  }).then(resp => resp.json()).then(data => {
    currentTaskId = data.task_id;
    connectSSE(data.task_id);
  }).catch(e => {
    showError(e.message);
    resetButton();
  });
}

// ============================================================
// SSE 连接
// ============================================================
function connectSSE(taskId) {
  if (eventSource) eventSource.close();

  eventSource = new EventSource(`/api/research/${taskId}/stream`);

  eventSource.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleProgress(msg);
  };

  eventSource.onerror = () => {
    // SSE 连接结束时检查结果
    eventSource.close();
    if (currentTaskId) fetchResult();
  };
}

// ============================================================
// 处理进度事件
// ============================================================
function handleProgress(msg) {
  switch (msg.type) {
    case 'phase':
      document.getElementById('phaseText').textContent =
        `${msg.status === 'running' ? '⏳' : '✅'} Phase ${msg.phase}: ${msg.label}`;
      if (msg.models_found) {
        document.getElementById('phaseText').textContent +=
          ` — 发现 ${msg.models_found} 个心智模型`;
      }
      break;

    case 'agent_start':
      updateAgentCard(msg.agent_id, 'running', '进行中...');
      break;

    case 'agent_done':
      if (msg.has_error) {
        updateAgentCard(msg.agent_id, 'error', `出错: ${msg.error}`);
      } else {
        updateAgentCard(msg.agent_id, 'done', `${msg.sources_count} 个来源`);
      }
      break;

    case 'model_found':
      // 显示发现的心智模型
      const grid = document.getElementById('agentGrid');
      const existingToast = document.getElementById('model-toast');
      if (existingToast) existingToast.remove();

      const toast = document.createElement('div');
      toast.id = 'model-toast';
      toast.style.cssText = `
        background: var(--surface2); border: 1px solid var(--accent);
        border-radius: var(--radius-sm); padding: 12px 16px; margin-top: 12px;
        font-size: 14px; animation: fadeIn 0.3s;
      `;
      toast.innerHTML = `🧠 <strong>发现心智模型:</strong> ${msg.name} — ${msg.description}`;
      grid.parentElement.insertBefore(toast, grid.nextSibling);
      setTimeout(() => toast.remove(), 5000);
      break;

    case 'error':
      showError(msg.message);
      resetButton();
      break;

    case 'complete':
      document.getElementById('phaseText').textContent = '✅ 蒸馏完成！';
      document.querySelector('.spinner').style.display = 'none';
      break;
  }
}

function updateAgentCard(agentId, status, statusText) {
  const card = document.getElementById(`agent-${agentId}`);
  if (!card) return;

  card.className = `agent-card ${status}`;
  const statusEl = card.querySelector('.agent-status');
  if (statusEl) statusEl.textContent = statusText;

  if (status === 'running') {
    card.querySelector('.agent-status').innerHTML = '<span class="spinner" style="width:12px;height:12px;display:inline-block;vertical-align:middle;margin-right:4px;"></span> 进行中...';
  }
}

// ============================================================
// 获取最终结果
// ============================================================
async function fetchResult() {
  if (!currentTaskId) return;

  try {
    const resp = await fetch(`/api/research/${currentTaskId}/result`);
    if (!resp.ok) throw new Error('获取结果失败');

    const result = await resp.json();
    renderResults(result);
  } catch(e) {
    showError(e.message);
  } finally {
    resetButton();
  }
}

// ============================================================
// 渲染结果
// ============================================================
function renderResults(profile) {
  const area = document.getElementById('resultsArea');
  area.classList.add('active');
  document.getElementById('progressArea').classList.remove('active');

  let html = '';

  // Quality badge
  const quality = profile.quality || {};
  html += `<div style="margin-bottom: 20px; display: flex; align-items: center; gap: 12px;">
    <span class="quality-badge ${quality.verdict || 'unknown'}">${quality.verdict === 'pass' ? '✅' : '⚠️'} 质量: ${quality.overall || '?'}/10</span>
    <span style="font-size:13px;color:var(--text2);">为「${e(profile.subject_name)}」生成的认知画像</span>
  </div>`;

  // One-liner
  if (profile.one_liner) {
    html += `<div class="one-liner">💡 ${e(profile.one_liner)}</div>`;
  }

  // === Expression DNA ===
  const dna = profile.expression_dna || {};
  html += `<div class="section-card">
    <div class="section-title"><span class="icon">🧬</span> 表达 DNA</div>
    <div class="dna-grid">
      <div class="dna-item"><div class="dna-label">语气特征</div><div class="dna-value">${e(dna.voice_character || '未识别')}</div></div>
      <div class="dna-item"><div class="dna-label">句式风格</div><div class="dna-value">${e(dna.sentence_style || '未识别')}</div></div>
      <div class="dna-item"><div class="dna-label">幽默类型</div><div class="dna-value">${e(dna.humor_type || '未识别')}</div></div>
      <div class="dna-item"><div class="dna-label">确定性模式</div><div class="dna-value">${e(dna.certainty_pattern || '未识别')}</div></div>
    </div>`;

  if (dna.signature_phrases && dna.signature_phrases.length > 0) {
    html += '<div class="signature-phrases">';
    for (const p of dna.signature_phrases) {
      html += `<span class="phrase-tag">${e(p)}</span>`;
    }
    html += '</div>';
  }
  html += '</div>';

  // === Mental Models ===
  const models = profile.mental_models || [];
  if (models.length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">🧠</span> 心智模型 (${models.length})</div>
      <div class="model-cards">`;
    for (const m of models) {
      html += `<div class="model-card">
        <div class="model-name">${e(m.name)}</div>
        <div class="model-desc">${e(m.description)}</div>
        <div class="model-details">
          ${m.evidence && m.evidence.length > 0 ? '<ul>' + m.evidence.slice(0,3).map(ev => `<li>${e(ev)}</li>`).join('') + '</ul>' : ''}
        </div>`;
      if (m.limitations && m.limitations.length > 0) {
        html += `<div class="model-limitations">
          <div class="limit-title">⚠️ 失效条件</div>
          ${m.limitations.map(l => `<div style="font-size:12px;color:var(--text2);">• ${e(l)}</div>`).join('')}
        </div>`;
      }
      html += '</div>';
    }
    html += '</div></div>';
  }

  // === Decision Heuristics ===
  const heuristics = profile.decision_heuristics || [];
  if (heuristics.length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">⚖️</span> 决策启发式 (${heuristics.length})</div>
      <div class="heuristic-list">`;
    for (const h of heuristics) {
      html += `<div class="heuristic-item">
        <div class="heuristic-rule">若 ${e(h.condition)} → 则 ${e(h.action)}</div>
        ${h.context ? `<div class="heuristic-context">出处: ${e(h.context)}</div>` : ''}
      </div>`;
    }
    html += '</div></div>';
  }

  // === Timeline ===
  const timeline = profile.timeline || [];
  if (timeline.length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">📅</span> 关键时间线</div>
      <div class="timeline-visual">`;
    for (const t of timeline.slice(0, 10)) {
      html += `<div class="timeline-item">
        <div class="timeline-year">${e(t.year)}</div>
        <div class="timeline-event">${e(t.event)}</div>
        ${t.significance ? `<div class="timeline-sig">${e(t.significance)}</div>` : ''}
      </div>`;
    }
    html += '</div></div>';
  }

  // === Anti-patterns & Honesty Boundaries (side by side) ===
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">';
  if ((profile.anti_patterns || []).length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">🚫</span> 反模式 / 价值观底线</div>
      <div class="list-items">`;
    for (const ap of profile.anti_patterns) {
      html += `<div class="list-item"><span class="bullet red"></span><span>${e(ap)}</span></div>`;
    }
    html += '</div></div>';
  }
  if ((profile.honesty_boundaries || []).length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">⚠️</span> 诚实边界</div>
      <div class="list-items">`;
    for (const b of profile.honesty_boundaries) {
      html += `<div class="list-item"><span class="bullet yellow"></span><span>${e(b)}</span></div>`;
    }
    html += '</div></div>';
  }
  html += '</div>';

  // === Internal Tensions ===
  const tensions = profile.internal_tensions || [];
  if (tensions.length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">⚡</span> 内在张力</div>`;
    for (const t of tensions) {
      html += `<div class="tension-item">🔀 ${e(t)}</div>`;
    }
    html += '</div>';
  }

  // === Source Summary ===
  if (profile.source_summary && Object.keys(profile.source_summary).length > 0) {
    html += `<div class="section-card">
      <div class="section-title"><span class="icon">📊</span> 研究来源分布</div>
      <div class="source-grid">`;
    for (const [name, count] of Object.entries(profile.source_summary)) {
      html += `<div class="source-stat"><div class="count">${count}</div><div class="label">${e(name)}</div></div>`;
    }
    html += '</div>';
  }

  // === All Sources with Links ===
  if (profile.all_sources && profile.all_sources.length > 0) {
    html += `<details style="margin-top:16px;" open>
      <summary style="cursor:pointer;font-size:14px;color:var(--accent);font-weight:600;">
        🔗 全部数据来源 (${profile.all_sources.length} 条) — 点击展开/收起
      </summary>
      <div style="margin-top:12px;max-height:500px;overflow-y:auto;font-size:12px;">`;

    // Group sources by agent
    const grouped = {};
    for (const s of profile.all_sources) {
      const key = (s.agent_icon || '') + ' ' + (s.agent_name || 'Unknown');
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(s);
    }

    for (const [agent, sources] of Object.entries(grouped)) {
      html += `<div style="margin-bottom:12px;">
        <div style="font-weight:600;color:var(--text);margin-bottom:4px;">${e(agent)} (${sources.length})</div>`;
      for (const s of sources) {
        if (s.url) {
          html += `<div style="padding:3px 0 3px 12px;border-left:2px solid var(--border);margin-bottom:2px;">
            <a href="${e(s.url)}" target="_blank" rel="noopener" style="color:var(--blue);text-decoration:none;">${e(s.title || s.url)}</a>
            <span style="color:var(--text2);margin-left:6px;">${e(s.snippet || '').substring(0, 100)}</span>
          </div>`;
        } else {
          html += `<div style="padding:3px 0 3px 12px;border-left:2px solid var(--border);color:var(--text2);">${e(s.title)}</div>`;
        }
      }
      html += '</div>';
    }

    html += '</div></details>';
  }

  html += '</div>';

  // Quality issues
  if (quality.issues && quality.issues.length > 0) {
    html += `<div class="section-card" style="border-color:rgba(251,191,36,0.3);">
      <div class="section-title"><span class="icon">🔍</span> 质量审查发现</div>
      ${quality.issues.map(i => `<div style="font-size:13px;color:var(--yellow);padding:4px 0;">• ${e(i)}</div>`).join('')}
    </div>`;
  }

  area.innerHTML = html;
  area.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// 工具函数
// ============================================================
function e(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  banner.textContent = '❌ ' + msg;
  banner.style.display = 'block';
}

function resetButton() {
  const btn = document.getElementById('distillBtn');
  btn.disabled = false;
  btn.textContent = '⚡ 开始蒸馏';
}
</script>
</body>
</html>"""


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    missing = check_config()
    if missing:
        print(f"\n  [WARN] Missing API Key config: {', '.join(missing)}")
        print(f"  Copy .env.example to .env and fill in your keys\n")
        print(f"    DEEPSEEK_API_KEY -> https://platform.deepseek.com/")
        print(f"    TAVILY_API_KEY   -> https://app.tavily.com/ (free 1000/month)\n")

    print("  Nuwa Web starting...")
    print("  Open: http://localhost:8000\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
