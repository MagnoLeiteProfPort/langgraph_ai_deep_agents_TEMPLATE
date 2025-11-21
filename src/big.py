import logging
from datetime import datetime
import warnings

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from deep_agents_libs.config.settings import get_settings
from deep_agents_libs.file_tools import ls, read_file, write_file
from deep_agents_libs.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_agents_libs.research_tools import (
    tavily_search,
    think_tool,
    get_today_str,
)
from deep_agents_libs.state import DeepAgentState
from deep_agents_libs.task_tool import _create_task_tool
from deep_agents_libs.todo_tools import write_todos, read_todos

from utils import show_prompt, stream_agent, format_messages

# ---------------------------------------------------------------------------
# Warning filters (cleaner console, but keeps behavior the same)
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="create_react_agent has been moved to `langchain.agents`",
)
warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7 for run and trace identifiers.",
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load settings from .env and configure logging
# ---------------------------------------------------------------------------
settings = get_settings()
OPEN_API_KEY = settings.openai_api_key

logger.info("Application starting with environment=%s debug=%s", settings.environment, settings.debug)
logger.debug("OpenAI API key loaded (value hidden); database_url=%s", settings.database_url)

# ---------------------------------------------------------------------------
# Model and tools setup
# ---------------------------------------------------------------------------
logger.info("Initializing main LLM model for agent")
model = init_chat_model(
    model="anthropic:claude-sonnet-4-20250514",
    temperature=0.0,
)
logger.debug("Model initialized: anthropic:claude-sonnet-4-20250514")

max_concurrent_research_units = 3
max_researcher_iterations = 3
logger.debug(
    "Research configuration: max_concurrent_research_units=%d, max_researcher_iterations=%d",
    max_concurrent_research_units,
    max_researcher_iterations,
)

sub_agent_tools = [tavily_search, think_tool]
built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]
logger.debug(
    "Built-in tools configured: %s",
    [t.__name__ if hasattr(t, "__name__") else getattr(t, "name", str(t)) for t in built_in_tools],
)

research_sub_agent = {
    "name": "research-agent",
    "description": (
        "Delegate research to the sub-agent researcher. "
        "Only give this researcher one topic at a time."
    ),
    "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["tavily_search", "think_tool"],
}
logger.info("Configured research sub-agent: %s", research_sub_agent["name"])

task_tool = _create_task_tool(
    sub_agent_tools,
    [research_sub_agent],
    model,
    DeepAgentState,
)
logger.info("Task delegation tool created")

delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools
logger.debug(
    "Total tools available to main agent: %d", len(all_tools)
)

# ---------------------------------------------------------------------------
# Build main prompt
# ---------------------------------------------------------------------------
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=get_today_str(),
)

INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

logger.debug("Main system prompt constructed")
# show_prompt(INSTRUCTIONS)  # Uncomment for debugging the prompt

# ---------------------------------------------------------------------------
# Create agent
# ---------------------------------------------------------------------------
logger.info("Creating main LangGraph ReAct agent")
agent = create_react_agent(
    model,
    all_tools,
    prompt=INSTRUCTIONS,
    state_schema=DeepAgentState,
)
logger.info("Agent created successfully; generating graph visualization")

try:
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
    logger.debug("Agent graph visualization rendered")
except Exception:
    logger.exception("Failed to render agent graph visualization")

# Simple test user message (for initial debug)
user_message = (
    "Say hello, say ok and stop."
)
logger.info("Running sample invocation for debug purposes")
logger.debug("Sample user_message: %s", user_message)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ],
    },
    config={
        "recursion_limit": 50,
    },
)
logger.info("Sample invocation completed; formatting messages for display")
format_messages(result["messages"])
