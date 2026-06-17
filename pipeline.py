"""
女娲 Nuwa Web — 五阶段蒸馏流水线

Phase 2: 三重验证提炼 → 心智模型 + 决策启发式
Phase 3: 生成五层认知画像
Phase 4: 质量验证

支持 OpenAI 兼容接口 (DeepSeek / OpenAI / 硅基流动 等)
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from openai import AsyncOpenAI

import config
from agents import AgentResult, AGENT_DEFINITIONS


# ============================================================
# 数据结构
# ============================================================

@dataclass
class MentalModel:
    name: str
    description: str
    evidence: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class DecisionHeuristic:
    condition: str
    action: str
    context: str = ""


@dataclass
class ExpressionDNA:
    voice_character: str = ""
    signature_phrases: list[str] = field(default_factory=list)
    sentence_style: str = ""
    humor_type: str = ""
    certainty_pattern: str = ""


@dataclass
class TimelineEvent:
    year: str
    event: str
    significance: str = ""


@dataclass
class CognitiveProfile:
    subject_name: str = ""
    generated_at: str = ""
    expression_dna: ExpressionDNA = field(default_factory=ExpressionDNA)
    mental_models: list[MentalModel] = field(default_factory=list)
    decision_heuristics: list[DecisionHeuristic] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    honesty_boundaries: list[str] = field(default_factory=list)
    internal_tensions: list[str] = field(default_factory=list)
    source_summary: dict[str, int] = field(default_factory=dict)
    all_sources: list[dict] = field(default_factory=list)  # [{agent_name, title, url, snippet}]
    one_liner: str = ""
    _raw: dict = field(default_factory=dict)


# ============================================================
# Phase 2-3: 三重验证提炼 → 生成认知画像
# ============================================================

TRIPLE_VERIFICATION_PROMPT = """你是一位认知科学家，专门从大量信息中提炼一个人的心智模型。

以下是对「{name}」的六维度研究结果。请从此人的思维方式和决策模式中提取核心框架。

## 三重验证标准

一个论点要成为「心智模型」，必须同时满足：
1. **跨域复现**: 在 >=2 个不同领域/场景中出现过
2. **生成力（预测力）**: 能用此模型推断此人对新问题的立场
3. **排他性**: 不是所有聪明人都这么想（区别于普适智慧）

- 通过 3 重的 → 心智模型
- 通过 1-2 重的 → 决策启发式
- 0 重 → 丢弃

## 输出格式

请严格用以下 JSON 格式回复：

{{
  "mental_models": [
    {{
      "name": "模型名称",
      "description": "一句话核心描述",
      "evidence": ["跨域证据1", "证据2"],
      "applications": ["应用场景1", "场景2"],
      "limitations": ["失效条件：什么情况下这个模型不管用"],
      "verification": {{"cross_domain": true, "predictive": true, "exclusive": true}}
    }}
  ],
  "decision_heuristics": [
    {{
      "condition": "当 X 时",
      "action": "则 Y",
      "context": "出处/场景"
    }}
  ],
  "expression_dna": {{
    "voice_character": "语气特征",
    "signature_phrases": ["标志短语1"],
    "sentence_style": "句式风格",
    "humor_type": "幽默类型",
    "certainty_pattern": "确定性表达模式"
  }},
  "anti_patterns": ["此人明确不做/反对的事"],
  "internal_tensions": ["内在矛盾1"],
  "honesty_boundaries": ["此认知画像的局限1"],
  "one_liner": "如果只能用一句话概括此人的思维操作系统，会是..."
}}

## 研究材料

{research_data}"""


async def synthesize_cognitive_profile(
    target_name: str,
    agent_results: dict[str, AgentResult],
    llm_client: AsyncOpenAI,
    progress_queue: asyncio.Queue,
) -> CognitiveProfile:
    """Phase 2-3: 三重验证提炼 → 生成认知画像"""
    await progress_queue.put({
        "type": "phase", "phase": 2, "label": "三重验证提炼", "status": "running"
    })

    # 组装研究材料
    research_parts = []
    for agent_id, result in agent_results.items():
        defn = AGENT_DEFINITIONS[agent_id]
        section = f"### {defn['icon']} {defn['name']}\n\n"

        if result.error:
            section += f"[注意] 此维度采集出错: {result.error}\n"
        else:
            section += f"来源数: {len(result.sources)}\n"
            for s in result.sources[:5]:
                section += f"- {s['title']}: {s['snippet'][:200]}\n"

            if isinstance(result.key_findings, dict) and "error" not in result.key_findings:
                section += f"\n提取结果:\n```json\n{json.dumps(result.key_findings, ensure_ascii=False, indent=2)[:3000]}\n```\n"
            elif result.raw_content:
                section += f"\n原始内容摘要:\n{result.raw_content[:2000]}\n"

        research_parts.append(section)

    research_data = "\n\n---\n\n".join(research_parts)

    try:
        response = await llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=8192,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "你是一位认知科学家。始终以有效 JSON 格式回复，不要输出其他内容。"},
                {"role": "user", "content": TRIPLE_VERIFICATION_PROMPT.format(
                    name=target_name,
                    research_data=research_data[:config.LLM_CONTEXT_LIMIT]
                )}
            ]
        )

        text = response.choices[0].message.content
        synthesis = _parse_json(text)

    except Exception as e:
        synthesis = {"error": str(e), "mental_models": [], "decision_heuristics": []}

    await progress_queue.put({
        "type": "phase", "phase": 2, "label": "三重验证提炼", "status": "completed",
        "models_found": len(synthesis.get("mental_models", [])),
        "heuristics_found": len(synthesis.get("decision_heuristics", [])),
    })

    for model in synthesis.get("mental_models", [])[:7]:
        await progress_queue.put({
            "type": "model_found",
            "name": model.get("name", ""),
            "description": model.get("description", ""),
        })

    # Phase 3: 组装认知画像
    await progress_queue.put({
        "type": "phase", "phase": 3, "label": "生成认知画像", "status": "running"
    })

    profile = CognitiveProfile(
        subject_name=target_name,
        generated_at="",
        mental_models=[
            MentalModel(
                name=m.get("name", ""),
                description=m.get("description", ""),
                evidence=m.get("evidence", []),
                applications=m.get("applications", []),
                limitations=m.get("limitations", []),
            )
            for m in synthesis.get("mental_models", [])[:config.MAX_MENTAL_MODELS]
        ],
        decision_heuristics=[
            DecisionHeuristic(
                condition=h.get("condition", ""),
                action=h.get("action", ""),
                context=h.get("context", ""),
            )
            for h in synthesis.get("decision_heuristics", [])[:config.MAX_HEURISTICS]
        ],
        expression_dna=ExpressionDNA(
            voice_character=synthesis.get("expression_dna", {}).get("voice_character", ""),
            signature_phrases=synthesis.get("expression_dna", {}).get("signature_phrases", []),
            sentence_style=synthesis.get("expression_dna", {}).get("sentence_style", ""),
            humor_type=synthesis.get("expression_dna", {}).get("humor_type", ""),
            certainty_pattern=synthesis.get("expression_dna", {}).get("certainty_pattern", ""),
        ),
        anti_patterns=synthesis.get("anti_patterns", []),
        internal_tensions=synthesis.get("internal_tensions", []),
        honesty_boundaries=synthesis.get("honesty_boundaries", []),
        one_liner=synthesis.get("one_liner", ""),
        source_summary={
            AGENT_DEFINITIONS[aid]["name"]: len(res.sources)
            for aid, res in agent_results.items()
        },
        all_sources=[
            {
                "agent_name": AGENT_DEFINITIONS[aid]["name"],
                "agent_icon": AGENT_DEFINITIONS[aid]["icon"],
                "title": s["title"],
                "url": s["url"],
                "snippet": s["snippet"][:200],
            }
            for aid, res in agent_results.items()
            for s in res.sources
        ],
        _raw={
            "agent_ids": list(agent_results.keys()),
            "errors": {aid: res.error for aid, res in agent_results.items() if res.error}
        }
    )

    await progress_queue.put({
        "type": "phase", "phase": 3, "label": "生成认知画像", "status": "completed",
        "models_count": len(profile.mental_models),
        "heuristics_count": len(profile.decision_heuristics),
    })

    return profile


# ============================================================
# Phase 4: 质量验证
# ============================================================

def _build_quality_prompt(profile: CognitiveProfile) -> str:
    return f"""你是一位质量控制专家。请评估以下对「{profile.subject_name}」的认知画像的质量。

## 评估维度

1. **心智模型数量**: 应有 3-7 个 (当前: {len(profile.mental_models)})
2. **决策启发式数量**: 应有 5-10 条 (当前: {len(profile.decision_heuristics)})
3. **表达DNA辨识度**: 读 100 字能否认出此人？
4. **诚实边界**: 是否 >=3 条具体局限？
5. **内在张力**: 是否 >=2 对矛盾？
6. **失效条件**: 每个模型是否写明不适用场景？

## 认知画像

```json
{json.dumps(asdict(profile), ensure_ascii=False, indent=2)[:20000]}
```

请用 JSON 回复评估结果：
{{
  "scores": {{
    "model_quantity": {{"score": 1-10, "note": ""}},
    "heuristic_quantity": {{"score": 1-10, "note": ""}},
    "expression_recognizability": {{"score": 1-10, "note": ""}},
    "honesty": {{"score": 1-10, "note": ""}},
    "internal_tension": {{"score": 1-10, "note": ""}},
    "failure_conditions": {{"score": 1-10, "note": ""}}
  }},
  "overall": 1-10,
  "issues": ["需要改进的问题"],
  "verdict": "pass / needs_revision / fail"
}}"""


async def run_quality_check(
    profile: CognitiveProfile,
    llm_client: AsyncOpenAI,
    progress_queue: asyncio.Queue,
) -> dict:
    """Phase 4: 质量验证"""
    await progress_queue.put({
        "type": "phase", "phase": 4, "label": "质量验证", "status": "running"
    })

    try:
        response = await llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=2048,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "你是质量控制专家。始终以 JSON 格式回复。"},
                {"role": "user", "content": _build_quality_prompt(profile)}
            ]
        )

        text = response.choices[0].message.content
        quality = _parse_json(text)
    except Exception as e:
        quality = {"error": str(e), "verdict": "unknown"}

    await progress_queue.put({
        "type": "phase", "phase": 4, "label": "质量验证", "status": "completed",
        "verdict": quality.get("verdict", "unknown"),
        "overall_score": quality.get("overall", 0),
        "issues": quality.get("issues", []),
    })

    return quality


# ============================================================
# 工具函数
# ============================================================

def _parse_json(text: str) -> dict:
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


def profile_to_dict(profile: CognitiveProfile) -> dict:
    """将 CognitiveProfile 转为可序列化的字典"""
    return {
        "subject_name": profile.subject_name,
        "generated_at": profile.generated_at,
        "one_liner": profile.one_liner,
        "expression_dna": {
            "voice_character": profile.expression_dna.voice_character,
            "signature_phrases": profile.expression_dna.signature_phrases,
            "sentence_style": profile.expression_dna.sentence_style,
            "humor_type": profile.expression_dna.humor_type,
            "certainty_pattern": profile.expression_dna.certainty_pattern,
        },
        "mental_models": [
            {
                "name": m.name,
                "description": m.description,
                "evidence": m.evidence,
                "applications": m.applications,
                "limitations": m.limitations,
            }
            for m in profile.mental_models
        ],
        "decision_heuristics": [
            {"condition": h.condition, "action": h.action, "context": h.context}
            for h in profile.decision_heuristics
        ],
        "anti_patterns": profile.anti_patterns,
        "timeline": [
            {"year": t.year, "event": t.event, "significance": t.significance}
            for t in profile.timeline
        ],
        "honesty_boundaries": profile.honesty_boundaries,
        "internal_tensions": profile.internal_tensions,
        "source_summary": profile.source_summary,
        "all_sources": profile.all_sources,
    }
