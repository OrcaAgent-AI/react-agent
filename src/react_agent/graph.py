"""ReAct agent with intelligent tool matching."""

import logging
from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from orcakit_sdk.utils import get_message_text, load_chat_model
from pydantic import BaseModel, Field

from react_agent.context import Context, get_context
from react_agent.prompts import REFUSAL_RESPONSE_PROMPT, TOOL_MATCHING_PROMPT
from react_agent.state import InputState, State
from react_agent.tools import get_tools

logger = logging.getLogger(__name__)


def _find_last_human_message(messages):
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg
    return None


async def tool_matcher(
    state: State, config: RunnableConfig, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Match appropriate tools based on user query."""
    if not state.messages:
        return {"match_tools": []}

    latest_human_message = _find_last_human_message(state.messages)
    if latest_human_message is None:
        return {"match_tools": []}

    available_tools = await get_tools(config)
    tool_info = [
        (
            getattr(tool, "name", getattr(tool, "__name__", str(tool))),
            getattr(tool, "description", "") or getattr(tool, "__doc__", "") or "",
        )
        for tool in available_tools
    ]

    if not tool_info:
        return {"match_tools": []}

    class ToolSelection(BaseModel):
        match_tools: list[str] = Field(
            description="List of tool names relevant to the user's query"
        )

    user_text = get_message_text(latest_human_message)
    tools_description = "\n".join([f"- {name}: {desc}" for name, desc in tool_info])
    tool_selection_prompt = TOOL_MATCHING_PROMPT.format(
        user_text=user_text, tools_description=tools_description
    )

    try:
        model_name = (
            config.get("configurable", {}).get("model", "")
            or get_context(runtime).model
        )
        model = load_chat_model(model_name)
        structured_model = model.with_structured_output(ToolSelection)
        response = await structured_model.ainvoke(
            [{"role": "user", "content": tool_selection_prompt}]
        )

        available_tool_names = {name for name, _ in tool_info}
        validated_tools = [
            tool for tool in response.match_tools if tool in available_tool_names
        ]
        return (
            {"match_tools": validated_tools} if validated_tools else {"match_tools": []}
        )

    except Exception as e:
        logger.warning("Tool matching failed: %s", e)
        return {"match_tools": [name for name, _ in tool_info]}


async def refuse_answer(
    state: State, config: RunnableConfig, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    """Generate refusal message when no appropriate tools are available."""
    available_tools = await get_tools(config)
    tool_names = [
        getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        for tool in available_tools
    ]
    tool_descriptions = [getattr(tool, "description", "") for tool in available_tools]
    capability_info = "\n".join(
        [f"- {name}: {desc}" for name, desc in zip(tool_names, tool_descriptions)]
    )

    user_question = ""
    latest_human_message = _find_last_human_message(state.messages)
    if latest_human_message:
        user_question = get_message_text(latest_human_message)

    refusal_prompt = REFUSAL_RESPONSE_PROMPT.format(
        user_question=user_question, capability_info=capability_info
    )

    model_name = (
        config.get("configurable", {}).get("model", "") or get_context(runtime).model
    )
    model = load_chat_model(model_name)
    response = await model.ainvoke([{"role": "user", "content": refusal_prompt}])

    return {"messages": [cast(AIMessage, response)]}


async def call_model(
    state: State, config: RunnableConfig, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    """Execute LLM call with filtered tools."""
    available_tools = await get_tools(config)

    if state.match_tools:
        matched_tool_names = set(state.match_tools)
        filtered_tools = [
            tool
            for tool in available_tools
            if getattr(tool, "name", getattr(tool, "__name__", str(tool)))
            in matched_tool_names
        ]
    else:
        filtered_tools = available_tools

    model_name = (
        config.get("configurable", {}).get("model", "") or get_context(runtime).model
    )
    model = load_chat_model(model_name).bind_tools(filtered_tools)

    system_prompt = (
        config.get("configurable", {}).get("system_prompt", "")
        or get_context(runtime).system_prompt
    )
    current_time = datetime.now(tz=UTC).isoformat()
    system_message = system_prompt.format(
        system_time=current_time, current_time=current_time
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response]}


async def dynamic_tools_node(
    state: State, config: RunnableConfig, runtime: Runtime[Context]
) -> dict[str, list[ToolMessage]]:
    """Execute tool calls from the last AI message."""
    available_tools = await get_tools(config)
    tool_node = ToolNode(available_tools, handle_tool_errors=True)
    result = await tool_node.ainvoke(state)
    return cast(dict[str, list[ToolMessage]], result)


def create_graph(name: str | None = None):
    """Create the ReAct agent graph."""
    builder = StateGraph(State, input_schema=InputState, context_schema=Context)

    builder.add_node("tool_matcher", tool_matcher)
    builder.add_node("refuse_answer", refuse_answer)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", dynamic_tools_node)

    builder.add_edge(START, "tool_matcher")

    def route_tool_only_check(
        state: State, config: RunnableConfig, runtime: Runtime[Context]
    ) -> Literal["refuse_answer", "call_model"]:
        configurable = config.get("configurable", {})
        tool_only = (
            configurable.get("tool_only") or get_context(runtime).tool_only or False
        )
        if tool_only and not state.match_tools:
            return "refuse_answer"
        return "call_model"

    builder.add_conditional_edges("tool_matcher", route_tool_only_check)
    builder.add_edge("refuse_answer", END)

    def route_model_output(state: State) -> Literal["__end__", "tools"]:
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
        return "tools" if last_message.tool_calls else END

    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "call_model")

    return builder.compile(name=name or "ReAct Agent")


graph = create_graph()
