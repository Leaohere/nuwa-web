"""
女娲 Nuwa Web — 六路并行研究 Agent

6 个独立 Agent，分别负责不同维度的信息采集与分析。
每个 Agent 封装: Tavily 搜索 → 内容抓取 → LLM 提取结构化发现

支持 OpenAI 兼容接口 (DeepSeek / OpenAI / 硅基流动 等)
"""

import asyncio
import json
import httpx
from typing import Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI

import config


# ============================================================
# Agent 数据结构
# ============================================================

@dataclass
class AgentResult:
    """单个 Agent 的研究结果"""
    agent_id: str
    agent_name: str
    sources: list[dict] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    raw_content: str = ""
    error: Optional[str] = None


# ============================================================
# Agent 定义：维度 Prompt 模板
# ============================================================

AGENT_DEFINITIONS = {
    "01-writings": {
        "name": "著作采集",
        "icon": "📚",
        "description": "书籍、长文、论文、Newsletter",
        "include_domains": [],
        "search_queries": [
            "{name} 书籍 著作 代表作",
            "{name} book writings published works",
            "{name} 长文 论文 深度文章 newsletter",
            "{name} 推荐书单 阅读 影响",
            "{name} book review analysis substack medium",
        ],
        "system_prompt": "你是一位深度研究分析师。始终以有效的 JSON 格式回复，不要输出其他内容。",
        "extract_prompt": """请从以下关于「{name}」的著作/长文相关信息中，提取关键发现。

重点关注：
1. **反复出现的核心论点**（在 >=3 个不同地方出现 = 真信念）
2. **自创术语/概念**（此人独创或大力推广的词汇）
3. **智识谱系**（此人受谁影响？推荐过什么书？思想源头）
4. **思维框架**（他/她用什么框架分析问题？）

用 JSON 格式回复：
{{
  "core_themes": ["反复出现的核心论点1", "核心论点2", ...],
  "coined_terms": [{{"term": "术语名", "meaning": "含义", "context": "出处"}}],
  "intellectual_influences": ["影响此人的思想家/书籍"],
  "thinking_frameworks": [{{"name": "框架名", "description": "描述", "application": "应用场景"}}],
  "notable_quotes": ["标志性金句1", "金句2"],
  "summary": "一句话总结此人的著作特征"
}}"""
    },

    "02-conversations": {
        "name": "对话采集",
        "icon": "🎙️",
        "description": "播客、长视频、深度采访、AMA",
        "include_domains": ["youtube.com"],
        "search_queries": [
            "{name} 访谈 采访 对话",
            "{name} podcast interview long-form conversation",
            "{name} 播客 AMA 深度对话",
            "{name} 演讲 talk 即兴问答",
            "{name} interview youtube podcast appearance",
        ],
        "system_prompt": "你是一位对话分析专家。始终以有效的 JSON 格式回复。",
        "extract_prompt": """请从以下关于「{name}」的访谈/对话信息中，提取关键发现。

重点关注：
1. **即兴类比**（被追问时用的比喻——反映真实思维方式）
2. **逻辑漏洞弥补方式**（被质疑时如何回应？）
3. **拒绝回答的边界**（什么问题坚决不答？）
4. **立场变化的瞬间**（有没有观点发生了转变？在什么情境下？）
5. **情绪触发点**（什么话题让他/她明显激动或回避？）

用 JSON 格式回复：
{{
  "improvised_analogies": ["即兴比喻1及上下文", "比喻2及上下文"],
  "debate_style": "辩论/回应风格描述",
  "boundaries": ["不愿讨论的话题1", "话题2"],
  "stance_shifts": [{{"from": "原来立场", "to": "新立场", "trigger": "触发因素"}}],
  "emotional_triggers": [{{"topic": "话题", "reaction": "反应描述"}}],
  "summary": "一句话总结此人的对话特征"
}}"""
    },

    "03-expression-dna": {
        "name": "表达DNA",
        "icon": "🧬",
        "description": "Twitter/X/微博/即刻/短文",
        "include_domains": ["twitter.com", "x.com"],
        "search_queries": [
            "{name} Twitter X 推文 社交媒体",
            "{name} site:twitter.com {name} opinion thoughts",
            "{name} 微博 即刻 社交动态 金句",
            "{name} social media posts short-form writing style",
            "{name} tweet viral thread {name} commentary",
        ],
        "system_prompt": "你是一位语言风格分析师。始终以有效的 JSON 格式回复。",
        "extract_prompt": """请从以下关于「{name}」的社交媒体/短文本信息中，提取表达DNA。

重点关注：
1. **高频用词**（此人最爱用的 5-10 个词/短语——这些是思维指纹）
2. **句式指纹**（长句还是短句？排比还是反问？逻辑严密还是跳跃？）
3. **确定性语气比例**（"一定/必然" vs "可能/也许/取决于" 的频率）
4. **幽默方式**（讽刺？自嘲？荒诞？冷幽默？还是基本不幽默？）
5. **争议立场**（公开表达过的、与主流不同的观点）

用 JSON 格式回复：
{{
  "high_frequency_words": [{{"word": "词", "context": "使用场景"}}],
  "sentence_fingerprint": "句式特征描述（50字以内）",
  "certainty_ratio": {{"certain": 60, "tentative": 40, "note": "估计比例"}},
  "humor_style": "幽默类型 / 不幽默",
  "controversial_stances": ["争议观点1", "观点2"],
  "signature_phrases": ["口头禅/标志性短语"],
  "summary": "如果读100字，如何辨认是此人写的？"
}}"""
    },

    "04-external-views": {
        "name": "他者视角",
        "icon": "👁️",
        "description": "深度分析、书评、批评、非官方传记",
        "include_domains": ["reddit.com", "medium.com"],
        "search_queries": [
            "{name} 深度分析 解读 评价",
            "{name} criticism critique review",
            "{name} reddit {name} discussion opinion",
            "{name} biography profile deep dive analysis",
            "{name} controversial criticism debate",
        ],
        "system_prompt": "你是一位批判性思维分析师。始终以有效的 JSON 格式回复。",
        "extract_prompt": """请从以下关于「{name}」的外部评价/批评信息中，提取关键发现。

重点关注：
1. **外部观察到的模式**（别人发现但本人可能没意识到的行为/思维模式）
2. **此人忽略的盲点**（反复被不同人指出的问题）
3. **与同行拉开差距的特质**（为什么是他/她，而不是其他类似的人？）
4. **最有洞察力的批评**（不是情绪化攻击，而是有道理的批评）

用 JSON 格式回复：
{{
  "observed_patterns": ["外部观察到的模式1", "模式2"],
  "blind_spots": [{{"issue": "盲点", "cited_by": "指出者", "evidence": "证据"}}],
  "differentiators": ["区别于同行的核心特质"],
  "insightful_criticisms": [{{"criticism": "批评内容", "validity": "有效性评估"}}],
  "reputation_summary": "外界评价的一句话总结"
}}"""
    },

    "05-decisions": {
        "name": "决策记录",
        "icon": "⚖️",
        "description": "重大商业/人生决策、争议行为",
        "include_domains": [],
        "search_queries": [
            "{name} 重大决策 选择 转折点",
            "{name} decision making business strategy",
            "{name} 争议决定 商业判断 投资逻辑",
            "{name} biggest decision career move",
            "{name} business decision analysis case study",
        ],
        "system_prompt": "你是一位决策分析专家。始终以有效的 JSON 格式回复。",
        "extract_prompt": """请从以下关于「{name}」的决策记录中，提取决策模式。

重点关注：
1. **决策逻辑权重**（此人做决策时，最看重什么？成本？速度？人才？长期？规模？）
2. **事后反思**（哪些决策后来被此人承认是错的？为什么？）
3. **言行一致/不一致案例**（说了什么 vs 实际做了什么）
4. **决策速度**（快速拍板还是深思熟虑？什么类型决策快/慢？）
5. **风险偏好**（冒险案例 vs 保守案例）

用 JSON 格式回复：
{{
  "decision_weights": [{{"factor": "决策因素", "weight": "重要程度说明"}}],
  "admitted_mistakes": [{{"decision": "决策", "reflection": "反思内容"}}],
  "say_do_gaps": [{{"said": "说过什么", "did": "实际做了什么"}}],
  "decision_speed": "决策速度特征描述",
  "risk_profile": "风险偏好描述",
  "summary": "一句话总结此人的决策风格"
}}"""
    },

    "06-timeline": {
        "name": "时间线",
        "icon": "📅",
        "description": "从出生到现在的完整轨迹",
        "include_domains": ["wikipedia.org"],
        "search_queries": [
            "{name} 生平 经历 成长历程",
            "{name} biography timeline life story",
            "{name} 创业历程 职业生涯 关键节点",
            "{name} early life education career milestones",
            "{name} wikipedia biography {name}",
        ],
        "system_prompt": "你是一位传记研究员。始终以有效的 JSON 格式回复。",
        "extract_prompt": """请从以下关于「{name}」的生平信息中，提取时间线。

重点关注：
1. **关键里程碑**（10-15 个最重要的时间节点）
2. **思想转折点**（什么事件导致此人想法发生重大变化？）
3. **最近 12 个月动态**（防止认知过时）

用 JSON 格式回复：
{{
  "milestones": [
    {{"year": "年份", "event": "事件描述", "significance": "为什么重要"}}
  ],
  "turning_points": [
    {{"year": "年份", "from": "转变前状态/观点", "to": "转变后", "catalyst": "催化事件"}}
  ],
  "recent_12_months": ["最近动态1", "动态2"],
  "era_labels": [{{"period": "时期（如2012-2015）", "label": "此阶段的标签"}}],
  "summary": "一句话概括此人的人生轨迹"
}}"""
    }
}


# ============================================================
# Agent 执行引擎
# ============================================================

class ResearchAgent:
    """单个研究 Agent —— 负责一个维度的信息采集与分析"""

    def __init__(self, agent_id: str, target_name: str, llm_client: AsyncOpenAI):
        self.agent_id = agent_id
        self.target_name = target_name
        self.definition = AGENT_DEFINITIONS[agent_id]
        self.client = llm_client

    async def search(self) -> list[dict]:
        """使用 Tavily 搜索"""
        all_results = []
        queries = [q.format(name=self.target_name) for q in self.definition["search_queries"]]

        async with httpx.AsyncClient(timeout=30) as http:
            for query in queries:  # 使用全部搜索词
                try:
                    body = {
                        "api_key": config.TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": config.MAX_SEARCH_RESULTS,
                        "include_answer": True,
                    }
                    # 优先纳入指定域名 (如 Twitter/X)
                    domains = self.definition.get("include_domains", [])
                    if domains:
                        body["include_domains"] = domains

                    resp = await http.post(
                        "https://api.tavily.com/search",
                        json=body,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for r in data.get("results", []):
                            all_results.append({
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "snippet": r.get("content", ""),
                            })
                        if data.get("answer"):
                            all_results.append({
                                "title": "AI 摘要",
                                "url": "",
                                "snippet": data["answer"],
                            })
                except Exception:
                    continue

        seen = set()
        unique = []
        for r in all_results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        return unique[:40]  # 最多返回 40 条去重结果

    async def fetch_content(self, urls: list[str]) -> str:
        """抓取部分 URL 的页面内容摘要"""
        contents = []
        async with httpx.AsyncClient(timeout=20) as http:
            for url in urls[:config.MAX_FETCH_URLS]:
                try:
                    resp = await http.post(
                        "https://api.tavily.com/extract",
                        json={
                            "api_key": config.TAVILY_API_KEY,
                            "urls": [url],
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("results", []):
                            raw = item.get("raw_content", "")[:config.MAX_FETCH_LENGTH]
                            contents.append(raw)
                except Exception:
                    continue

        return "\n\n---\n\n".join(contents)

    async def extract(self, context: str) -> dict:
        """用 LLM (OpenAI 兼容接口) 从研究材料中提取结构化发现"""
        prompt = self.definition["extract_prompt"].format(name=self.target_name)

        try:
            # 使用 await 调用异步 OpenAI 客户端 (支持 DeepSeek 等)
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=4096,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.definition["system_prompt"]},
                    {"role": "user", "content": f"{prompt}\n\n=== 研究材料 ===\n{context[:config.LLM_CONTEXT_LIMIT]}\n\n请用 JSON 格式回复你的分析结果。"}
                ]
            )

            text = response.choices[0].message.content
            return self._parse_json(text)
        except Exception as e:
            return {"error": str(e), "raw": context[:500]}

    def _parse_json(self, text: str) -> dict:
        """从 LLM 回复中提取 JSON"""
        text = text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in text:
                si = text.find(start_char)
                ei = text.rfind(end_char) + 1
                try:
                    return json.loads(text[si:ei])
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_response": text}

    async def run(self) -> AgentResult:
        """执行完整的 Agent 研究流程"""
        result = AgentResult(
            agent_id=self.agent_id,
            agent_name=self.definition["name"],
        )

        try:
            sources = await asyncio.wait_for(self.search(), timeout=45)
            result.sources = sources

            if not sources:
                result.key_findings = ["未找到相关信息"]
                return result

            urls = [s["url"] for s in sources if s["url"]]
            fetched = await asyncio.wait_for(self.fetch_content(urls), timeout=60)
            result.raw_content = fetched

            context = fetched if fetched else "\n".join(
                f"{s['title']}: {s['snippet']}" for s in sources
            )
            extraction = await asyncio.wait_for(self.extract(context), timeout=90)
            result.key_findings = extraction if isinstance(extraction, dict) else {"raw": str(extraction)}

        except asyncio.TimeoutError:
            result.error = f"Agent {self.definition['name']} 超时"
        except Exception as e:
            result.error = str(e)
            if not result.sources:
                result.key_findings = [f"研究过程中遇到错误: {e}"]

        return result


# ============================================================
# 并行执行入口
# ============================================================

async def run_all_agents(
    target_name: str,
    progress_queue: asyncio.Queue,
    llm_client: AsyncOpenAI,
) -> dict[str, AgentResult]:
    """并行执行全部 6 个 Agent，实时推送进度到 queue"""
    agent_ids = list(AGENT_DEFINITIONS.keys())

    async def run_single(aid: str) -> AgentResult:
        agent = ResearchAgent(aid, target_name, llm_client)

        await progress_queue.put({
            "type": "agent_start",
            "agent_id": aid,
            "agent_name": AGENT_DEFINITIONS[aid]["name"],
            "icon": AGENT_DEFINITIONS[aid]["icon"],
        })

        res = await agent.run()

        await progress_queue.put({
            "type": "agent_done",
            "agent_id": aid,
            "agent_name": AGENT_DEFINITIONS[aid]["name"],
            "icon": AGENT_DEFINITIONS[aid]["icon"],
            "sources_count": len(res.sources),
            "has_error": res.error is not None,
            "error": res.error,
        })

        return res

    tasks = [run_single(aid) for aid in agent_ids]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for aid, res in zip(agent_ids, results_list):
        if isinstance(res, Exception):
            results[aid] = AgentResult(
                agent_id=aid,
                agent_name=AGENT_DEFINITIONS[aid]["name"],
                error=str(res),
                key_findings=[f"Agent 执行异常: {res}"]
            )
        else:
            results[aid] = res

    return results
