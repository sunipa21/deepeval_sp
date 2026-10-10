"""
llm_config.py
=============
One place to choose the agent LLM and the judge LLM.

    AGENT_MODEL  — any LangChain "provider:model" string (default openai:gpt-4o-mini)
                   e.g. openai:gpt-4o, anthropic:claude-sonnet-4-5
    JUDGE_MODEL  — model name DeepEval uses for LLM-as-judge metrics (default gpt-4o)

Set them in .env to switch models without touching code.
"""

import os

from dotenv import load_dotenv

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")


def get_chat_model(model: str | None = None):
    """Build the agent chat model. temperature=0 keeps runs as repeatable as possible."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(model or AGENT_MODEL, temperature=0)
