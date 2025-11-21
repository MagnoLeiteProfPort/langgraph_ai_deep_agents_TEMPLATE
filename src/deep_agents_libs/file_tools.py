"""Virtual file system tools for agent state management."""

import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_libs.prompts import (
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)
from deep_agents_libs.state import DeepAgentState

logger = logging.getLogger(__name__)


@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    files = list(state.get("files", {}).keys())
    logger.debug("ls() called; %d files found: %s", len(files), files)
    return files


@tool(description=READ_FILE_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit."""
    logger.info(
        "read_file called with file_path=%s, offset=%d, limit=%d",
        file_path,
        offset,
        limit,
    )
    files = state.get("files", {})
    if file_path not in files:
        logger.warning("File not found in virtual filesystem: %s", file_path)
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        logger.debug("File exists but is empty: %s", file_path)
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        logger.warning(
            "Offset %d exceeds file length (%d lines) for file %s",
            offset,
            len(lines),
            file_path,
        )
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    logger.debug(
        "read_file returning %d lines for file %s starting at offset %d",
        len(result_lines),
        file_path,
        offset,
    )
    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file in the virtual filesystem."""
    logger.info("write_file called for path: %s", file_path)
    files = state.get("files", {})
    existing = file_path in files
    files[file_path] = content
    logger.debug(
        "File %s %s in virtual filesystem",
        file_path,
        "updated" if existing else "created",
    )
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )
