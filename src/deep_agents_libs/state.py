"""State management for deep agents with TODO tracking and virtual file systems."""

import logging
from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from langgraph.prebuilt.chat_agent_executor import AgentState

logger = logging.getLogger(__name__)


class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence."""
    logger.debug(
        "Merging file dictionaries in file_reducer: left_keys=%s, right_keys=%s",
        list((left or {}).keys()),
        list((right or {}).keys()),
    )
    if left is None:
        return right
    elif right is None:
        return left
    else:
        merged = {**left, **right}
        logger.debug("Merged files count: %d", len(merged))
        return merged


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system."""

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
