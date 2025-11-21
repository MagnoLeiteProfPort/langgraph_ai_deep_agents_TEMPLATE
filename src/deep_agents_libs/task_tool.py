"""Task delegation tools for context isolation through sub-agents."""

import logging
from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from deep_agents_libs.prompts import TASK_DESCRIPTION_PREFIX
from deep_agents_libs.state import DeepAgentState

logger = logging.getLogger(__name__)


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents."""
    logger.info(
        "Creating task delegation tool with %d subagents and %d tools",
        len(subagents),
        len(tools),
    )
    agents = {}

    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            logger.debug("Wrapping raw function as BaseTool: %s", getattr(tool_, "__name__", tool_))
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_
    logger.debug("Registered tools: %s", list(tools_by_name.keys()))

    for _agent in subagents:
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
            logger.debug(
                "Sub-agent '%s' uses specific tools: %s",
                _agent["name"],
                _agent["tools"],
            )
        else:
            _tools = tools
            logger.debug(
                "Sub-agent '%s' uses all available tools", _agent["name"]
            )
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )
        logger.info("Sub-agent created: %s", _agent["name"])

    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with isolated context."""
        logger.info(
            "task tool invoked for subagent_type=%s with description length=%d",
            subagent_type,
            len(description or ""),
        )

        if subagent_type not in agents:
            allowed = [f"`{k}`" for k in agents]
            logger.error(
                "Invalid subagent_type requested: %s; allowed types: %s",
                subagent_type,
                allowed,
            )
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {allowed}"

        sub_agent = agents[subagent_type]

        # Context isolation
        state["messages"] = [{"role": "user", "content": description}]
        logger.debug("Isolated state created for sub-agent '%s'", subagent_type)

        result = sub_agent.invoke(state)
        logger.info(
            "Sub-agent '%s' invocation completed; messages=%d",
            subagent_type,
            len(result.get("messages", [])),
        )

        return Command(
            update={
                "files": result.get("files", {}),
                "messages": [
                    ToolMessage(
                        result["messages"][-1].content,
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return task
