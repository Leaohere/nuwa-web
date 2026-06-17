"""
女娲 Nuwa Web — 配置管理
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置 (OpenAI 兼容接口)
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Tavily Search API (搜索用)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 搜索配置
MAX_SEARCH_RESULTS = 10       # 每次搜索返回条数 (原 5)
MAX_FETCH_URLS = 10           # 每个 Agent 抓取 URL 数 (原 5)
MAX_FETCH_LENGTH = 20000      # 单篇内容最大字符 (原 8000)
LLM_CONTEXT_LIMIT = 50000     # 送入 LLM 的上下文上限 (原 15000)

# 流水线配置
AGENT_TIMEOUT = 180           # 单个 Agent 超时秒数 (原 120)
PIPELINE_TIMEOUT = 900        # 整条流水线超时秒数 (原 600)

# 验证阈值
MIN_MENTAL_MODELS = 3
MAX_MENTAL_MODELS = 7
MIN_HEURISTICS = 5
MAX_HEURISTICS = 10
