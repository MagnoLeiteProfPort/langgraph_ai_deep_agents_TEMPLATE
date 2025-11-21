"""Research tools including web search and summarization utilities."""

import base64
import logging
import os
import uuid
from datetime import datetime
from typing_extensions import Annotated, Literal

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient

from deep_agents_libs.prompts import SUMMARIZE_WEB_SEARCH
from deep_agents_libs.state import DeepAgentState
from deep_agents_libs.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
OPEN_API_KEY = settings.openai_api_key
logger.debug("Loaded settings in research_tools with environment=%s", settings.environment)

# Summarization model 
summarization_model = init_chat_model(model="openai:gpt-4o-mini")
tavily_client = TavilyClient()


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def get_today_str() -> str:
    """Get current date in a human-readable, cross-platform format."""
    now = datetime.now()
    date_str = now.strftime("%a %b {}, %Y").format(now.day)
    logger.debug("get_today_str returning: %s", date_str)
    return date_str


def run_tavily_search(
    search_query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Perform search using Tavily API for a single query."""
    logger.info(
        "Running Tavily search: query=%s, max_results=%d, topic=%s, include_raw_content=%s",
        search_query,
        max_results,
        topic,
        include_raw_content,
    )
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    logger.debug("Tavily search completed; got %d top-level keys", len(result))
    return result


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model."""
    logger.debug(
        "Summarizing webpage content of length %d characters",
        len(webpage_content or ""),
    )
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_and_filename = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH.format(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )
        logger.info("Summarization completed successfully")
        return summary_and_filename
    except Exception:
        logger.exception("Error during summarization; falling back to basic summary")
        return Summary(
            filename="search_result.md",
            summary=(
                webpage_content[:1000] + "..."
                if len(webpage_content) > 1000
                else webpage_content
            ),
        )


def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content where available."""
    logger.debug("Processing search results")
    processed_results = []
    httpx_client = httpx.Client()

    for result in results.get("results", []):
        url = result["url"]
        logger.info("Fetching URL from Tavily result: %s", url)
        try:
            response = httpx_client.get(url, timeout=15)
        except Exception:
            logger.exception("HTTP error while fetching URL: %s", url)
            response = None

        if response is not None and response.status_code == 200:
            logger.debug("URL fetched successfully: %s", url)
            raw_content = markdownify(response.text)
            summary_obj = summarize_webpage_content(raw_content)
        else:
            logger.warning(
                "Failed to fetch URL or non-200 response (%s), using Tavily content for: %s",
                getattr(response, "status_code", "no-response"),
                url,
            )
            raw_content = result.get("raw_content", "")
            summary_obj = Summary(
                filename="URL_error.md",
                summary=result.get(
                    "content", "Error reading URL; try another search."
                ),
            )

        uid = (
            base64.urlsafe_b64encode(uuid.uuid4().bytes)
            .rstrip(b"=")
            .decode("ascii")[:8]
        )
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        processed_results.append(
            {
                "url": result["url"],
                "title": result["title"],
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": raw_content,
            }
        )

    logger.info(
        "Processed %d search result(s) into summarized form", len(processed_results)
    )
    return processed_results


@tool
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search web and save detailed results to files while returning minimal context."""
    logger.info(
        "tavily_search tool called with query=%s, max_results=%d, topic=%s",
        query,
        max_results,
        topic,
    )
    search_results = run_tavily_search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    processed_results = process_search_results(search_results)

    files = state.get("files", {})
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        filename = result["filename"]
        logger.debug("Saving processed result to virtual file: %s", filename)
        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""

        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")

    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

    logger.info(
        "tavily_search tool completed; %d file(s) saved: %s",
        len(saved_files),
        saved_files,
    )
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(summary_text, tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making."""
    logger.debug("think_tool called with reflection length %d", len(reflection))
    return f"Reflection recorded: {reflection}"
