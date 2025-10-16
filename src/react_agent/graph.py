"""Define a custom Reasoning and Action agent with intelligent tool matching.

This module implements a sophisticated ReAct (Reasoning and Acting) agent that uses
large language models for intelligent tool selection and execution. The agent features:

- LLM-based tool matching for query-appropriate tool selection
- Intelligent refusal mechanism when no suitable tools are available
- Dynamic tool filtering based on matched tools
- Comprehensive error handling and fallback mechanisms

The agent workflow follows this pattern:
1. Tool matching: Analyze user query and select appropriate tools
2. Conditional routing: Decide whether to refuse or proceed based on tool availability
3. Model execution: Use selected tools to answer user queries
4. Tool execution: Execute matched tools and return results

Key Features:
- Context-aware tool selection using structured LLM outputs
- Configurable tool-only mode for restricted capability responses
- MCP server integration for external tool providers
- Robust error handling with graceful degradation
"""

import logging
from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from common.utils import get_message_text, load_chat_model
from react_agent.context import Context
from react_agent.prompts import REFUSAL_RESPONSE_PROMPT, TOOL_MATCHING_PROMPT
from react_agent.state import InputState, State
from react_agent.tools import get_tools

logger = logging.getLogger(__name__)

# Define the function that calls the model


async def tool_matcher(state: State, runtime: Runtime[Context]) -> dict[str, list[str]]:
    """Match appropriate tools based on user query using LLM and update state.

    This function analyzes the user's latest query and uses a language model to
    intelligently select the most relevant tools from the available tool set.
    The selection is based on semantic understanding of the user's intent and
    the capabilities of each tool.

    Args:
        state (State): The current conversation state containing messages and metadata.
        runtime (Runtime[Context]): Runtime configuration with context settings.

    Returns:
        dict[str, list[str]]: A dictionary containing the 'match_tools' key with
            a list of tool names that are most relevant to the user's query.

    Raises:
        Exception: Logs warnings for LLM matching failures but provides fallback
            behavior to ensure system stability.

    Example:
        >>> result = await tool_matcher(state, runtime)
        >>> print(result["match_tools"])
        ['web_search', 'deepwiki_search']
    """
    # Get the latest user message (assuming it's the last message in the conversation)
    if not state.messages:
        return {"match_tools": []}

    latest_message = state.messages[-1]
    if not hasattr(latest_message, "type") or latest_message.type != "human":
        return {"match_tools": []}

    # Get available tools
    available_tools = await get_tools()

    # Extract tool names and descriptions for matching
    tool_info = []
    for tool in available_tools:
        tool_name = getattr(tool, "name", str(tool))
        tool_desc = getattr(tool, "description", "")
        tool_info.append((tool_name, tool_desc))

    if not tool_info:
        return {"match_tools": []}

    # Create structured output schema for tool selection
    class ToolSelection(BaseModel):
        match_tools: list[str] = Field(
            description="List of tool names that are most relevant to the user's query"
        )

    # Create tool selection prompt for LLM
    user_text = get_message_text(latest_message)
    tools_description = "\n".join([f"- {name}: {desc}" for name, desc in tool_info])

    tool_selection_prompt = TOOL_MATCHING_PROMPT.format(
        user_text=user_text, tools_description=tools_description
    )

    try:
        # Use LLM with structured output to select appropriate tools
        model = load_chat_model(runtime.context.model)
        structured_model = model.with_structured_output(ToolSelection)
        response = await structured_model.ainvoke(
            [{"role": "user", "content": tool_selection_prompt}]
        )

        # Validate that all selected tools exist in available tools
        available_tool_names = [name for name, _ in tool_info]
        validated_tools = [
            tool for tool in response.match_tools if tool in available_tool_names
        ]

        if validated_tools:
            return {"match_tools": validated_tools}
        else:
            # Fallback to all available tools if no valid tools selected
            return {"match_tools": []}

    except Exception as e:
        # Fallback to all available tools if LLM matching fails
        logger.warning(f"Tool matching failed: {e}")
        return {"match_tools": [tool_name for tool_name, _ in tool_info]}


async def refuse_answer(
    state: State, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    """Generate an intelligent refusal message when no appropriate tools are available.

    This function creates a comprehensive refusal response that explains why the agent
    cannot answer the user's question and provides detailed information about available
    capabilities. The response is generated using an LLM to ensure it's natural,
    helpful, and contextually appropriate.

    Args:
        state (State): The current conversation state containing the user's question.
        runtime (Runtime[Context]): Runtime configuration with context settings.

    Returns:
        dict[str, list[AIMessage]]: A dictionary containing the 'messages' key with
            a list containing a single AIMessage with the refusal response.

    Features:
        - Contextual refusal based on the specific user question
        - Detailed capability information listing all available tools
        - Helpful suggestions for how users can reformulate their queries
        - Professional and polite tone maintained through LLM generation

    Example:
        >>> result = await refuse_answer(state, runtime)
        >>> print(result["messages"][0].content)
        "I apologize, but I cannot help with cooking recipes. I can assist with
        web searches, knowledge base queries, and data analysis..."
    """
    # Get available tools information
    available_tools = await get_tools()
    tool_names = [getattr(tool, "name", str(tool)) for tool in available_tools]
    tool_descriptions = [getattr(tool, "description", "") for tool in available_tools]

    capability_info = "\n".join(
        [f"- {name}: {desc}" for name, desc in zip(tool_names, tool_descriptions)]
    )

    # Get the user's question from the latest message
    user_question = ""
    if state.messages:
        latest_message = state.messages[-1]
        if hasattr(latest_message, "type") and latest_message.type == "human":
            user_question = get_message_text(latest_message)

    # Create prompt for LLM to generate refusal message
    refusal_prompt = REFUSAL_RESPONSE_PROMPT.format(
        user_question=user_question, capability_info=capability_info
    )

    # Use LLM to generate refusal message
    model = load_chat_model(runtime.context.model)
    response = await model.ainvoke([{"role": "user", "content": refusal_prompt}])

    # Return the LLM-generated refusal message
    return {"messages": [cast(AIMessage, response)]}


async def call_model(
    state: State, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    """Execute the main LLM call with intelligent tool filtering and response processing.

    This function serves as the core reasoning component of the ReAct agent. It
    intelligently filters available tools based on the matched tools from the
    tool_matcher, initializes the language model with the appropriate tool set,
    and processes the model's response for further execution.

    Args:
        state (State): The current conversation state containing messages and
            matched tools information.
        runtime (Runtime[Context]): Runtime configuration containing model settings
            and system prompts.

    Returns:
        dict[str, list[AIMessage]]: A dictionary containing the 'messages' key with
            a list containing the model's response message(s).

    Features:
        - Dynamic tool filtering based on match_tools from tool_matcher
        - System prompt formatting with current timestamp
        - Tool binding for function calling capabilities
        - Graceful handling of recursion limits and step exhaustion
        - Comprehensive error handling with informative responses

    Processing Flow:
        1. Retrieve and filter available tools based on match_tools
        2. Initialize model with filtered tool set
        3. Format system prompt with current time
        4. Invoke model with conversation history
        5. Handle edge cases (recursion limits, tool call failures)
        6. Return structured response for graph processing

    Example:
        >>> result = await call_model(state, runtime)
        >>> response = result["messages"][0]
        >>> print(f"Model response: {response.content}")
        "Based on your query, I'll search for the latest information..."
    """
    # Get available tools based on configuration
    available_tools = await get_tools()

    # Filter tools based on match_tools if specified in state
    if state.match_tools:
        # Filter available tools to only include matched tools
        matched_tool_names = set(state.match_tools)
        filtered_tools = []
        for tool in available_tools:
            tool_name = getattr(tool, "name", str(tool))
            if tool_name in matched_tool_names:
                filtered_tools.append(tool)
        available_tools = filtered_tools

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context.model).bind_tools(available_tools)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def dynamic_tools_node(
    state: State, runtime: Runtime[Context]
) -> dict[str, list[ToolMessage]]:
    """Execute tools dynamically based on current configuration and state.

    This function serves as the action component of the ReAct agent. It retrieves
    the currently available tools (which may be filtered by previous nodes),
    creates a ToolNode with these tools, and executes the tool calls requested
    by the language model in the previous step.

    Args:
        state (State): The current conversation state containing messages with
            potential tool calls to execute.
        runtime (Runtime[Context]): Runtime configuration containing tool settings
            and execution parameters.

    Returns:
        dict[str, list[ToolMessage]]: A dictionary containing the 'messages' key with
            a list of ToolMessage objects representing the results of tool executions.

    Features:
        - Dynamic tool loading based on current configuration
        - Support for MCP (Model Context Protocol) tools
        - Automatic tool result formatting and error handling
        - Integration with LangGraph's ToolNode for reliable execution

    Execution Flow:
        1. Retrieve available tools from the tool registry
        2. Create a ToolNode with the current tool set
        3. Execute tool calls from the last AI message
        4. Return structured tool results for model processing

    Example:
        >>> result = await dynamic_tools_node(state, runtime)
        >>> tool_results = result["messages"]
        >>> for msg in tool_results:
        ...     print(f"Tool {msg.name}: {msg.content}")
    """
    # Get available tools based on configuration
    available_tools = await get_tools()

    # Create a ToolNode with the available tools
    tool_node = ToolNode(available_tools)

    # Execute the tool node
    result = await tool_node.ainvoke(state)

    return cast(dict[str, list[ToolMessage]], result)


# Define a new graph
# ====================
# This section builds the complete ReAct agent graph with intelligent tool matching
# and conditional routing capabilities.

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the nodes
# ================
# Each node represents a specific phase in the ReAct agent workflow:
builder.add_node("tool_matcher", tool_matcher)  # Phase 1: Intelligent tool selection
builder.add_node("refuse_answer", refuse_answer)  # Phase 2a: Capability-aware refusal
builder.add_node(
    "call_model", call_model
)  # Phase 2b: Core reasoning with filtered tools
builder.add_node("tools", dynamic_tools_node)  # Phase 3: Tool execution

# Set the entrypoint as `tool_matcher` to match tools first
# =========================================================
# All conversations start with tool matching to ensure context-appropriate responses
builder.add_edge(START, "tool_matcher")


def route_tool_only_check(
    state: State, runtime: Runtime[Context]
) -> Literal["refuse_answer", "call_model"]:
    """Route to appropriate node based on tool-only mode and tool availability.

    This routing function implements the core logic for the tool-only mode feature.
    It checks if the agent is configured to only use tools and whether appropriate
    tools have been matched for the user's query. Based on these conditions, it
    decides whether to proceed with normal processing or generate a refusal response.

    Args:
        state (State): The current conversation state containing matched tools
            information from the tool_matcher.
        runtime (Runtime[Context]): Runtime configuration containing the tool_only
            setting and other context parameters.

    Returns:
        Literal["refuse_answer", "call_model"]: The name of the next node to execute:
            - "refuse_answer": When tool_only=True and no tools were matched
            - "call_model": When tool_only=False or appropriate tools were found

    Decision Logic:
        - If runtime.context.tool_only is True AND state.match_tools is empty:
            → Route to "refuse_answer" to generate capability information
        - Otherwise: Route to "call_model" for normal processing

    Example:
        >>> # When tool_only=True and no tools matched
        >>> route = route_tool_only_check(state, runtime)
        >>> print(route)  # "refuse_answer"

        >>> # When tool_only=False or tools were matched
        >>> route = route_tool_only_check(state, runtime)
        >>> print(route)  # "call_model"
    """
    # Check if we should refuse to answer
    if runtime.context.tool_only and not state.match_tools:
        return "refuse_answer"

    # Otherwise, continue to call_model
    return "call_model"


# Add conditional edge from tool_matcher
# ======================================
# This implements the tool-only mode logic:
# - If tool_only=True and no tools matched → refuse_answer
# - Otherwise → call_model for normal processing
builder.add_conditional_edges(
    "tool_matcher",
    route_tool_only_check,
)

# Add edge from refuse_answer to end
# ==================================
# When refusing to answer, the conversation ends with capability information
builder.add_edge("refuse_answer", END)


def route_model_output(state: State) -> Literal[END, "tools"]:
    """Route to next node based on the model's response and tool call requirements.

    This routing function analyzes the last message from the language model to
    determine whether tool execution is required or if the conversation should
    end. It implements the core ReAct pattern by deciding between reasoning
    (model response) and acting (tool execution) phases.

    Args:
        state (State): The current conversation state containing the model's
            latest response message.

    Returns:
        Literal[END, "tools"]: The name of the next node to execute:
            - END: When the model provided a final answer without tool calls
            - "tools": When the model requested tool execution to gather information

    Decision Logic:
        - If the last message contains tool_calls: Route to "tools" for execution
        - If the last message has no tool_calls: Route to END (conversation complete)
        - Raises ValueError if the last message is not an AIMessage

    ReAct Pattern Implementation:
        - Reasoning: Model analyzes query and decides on actions (tool calls)
        - Acting: Tools are executed to gather information
        - Reflection: Results are fed back to the model for final response

    Example:
        >>> # Model requests tool execution
        >>> route = route_model_output(state_with_tool_calls)
        >>> print(route)  # "tools"

        >>> # Model provides final answer
        >>> route = route_model_output(state_with_final_answer)
        >>> print(route)  # END

    Raises:
        ValueError: If the last message in state.messages is not an AIMessage,
            indicating an unexpected message type in the conversation flow.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
# ====================================================================
# This implements the core ReAct pattern:
# - If model requests tools → route to "tools" for execution
# - If model provides final answer → route to END
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# ==============================================
# This creates the ReAct cycle: after using tools, we always return to the model
# for reflection and potential additional reasoning or final response generation
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# ============================================
# The final compiled graph represents the complete ReAct agent with:
# - Intelligent tool matching
# - Conditional refusal capabilities
# - Dynamic tool execution
# - Comprehensive error handling
graph = builder.compile(name="ReAct Agent")
