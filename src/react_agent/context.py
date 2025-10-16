"""Define the configurable parameters for the agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from react_agent.mcp_server_configs import MCP_SERVERS

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="compatible_openai/DeepSeek-V3-0324",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    tool_only: bool = field(
        default=False,
        metadata={
            "description": "Whether the agent should rely completely on tools for answering questions. "
            "When True, the agent will only use tools and not provide direct responses."
        },
    )

    enable_web_search: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable web search functionality. "
            "When False, web search tools will be disabled."
        },
    )

    mcp_server_configs: str = field(
        default=json.dumps(MCP_SERVERS, indent=2),
        metadata={
            "description": "JSON string containing MCP server configurations. "
            "This defines which MCP servers are available and their connection settings."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
