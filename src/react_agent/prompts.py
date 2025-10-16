"""Default prompts used by the agent."""

SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {system_time}"""

TOOL_MATCHING_PROMPT = """Based on the user's query, select the most appropriate tools from the available list.

User Query: "{user_text}"

Available Tools:
{tools_description}

Instructions:
1. Analyze the user's query to understand their intent
2. Select only the tools that are most relevant to answering their question
3. Consider the context and purpose of each tool
4. Return the selected tool names as a list

Select the most appropriate tools for this query."""

REFUSAL_RESPONSE_PROMPT = """The user asked the following question: "{user_question}"

However, this question is beyond my current capabilities because I cannot find appropriate tools to handle this question.

I can currently only help users through the following tools:

{capability_info}

Please generate a friendly and professional refusal response message with the following requirements:
1. Politely explain that I cannot answer the user's question
2. Explain the reason (no suitable tools available)
3. List my capability scope (the above tool list)
4. Suggest how the user can adjust their question to get help
5. Respond in Chinese

Please generate the refusal response:"""
