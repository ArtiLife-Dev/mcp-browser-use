import asyncio
import os
import sys
import traceback
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

from browser_use import BrowserConfig, Browser, Controller, Agent
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from browser_use import logging_config
from mcp.server.fastmcp import Context, FastMCP
from contextlib import redirect_stdout

from mcp_server_browser_use.utils import utils

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress unnecessary logging from libraries if needed
# logging.getLogger('browser_use').root.setLevel(logging.FATAL)
# logging_config.addLoggingLevel

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage the lifecycle of the Browser instance."""
    browser = None
    try:
        # Get browser configuration from environment variables once at startup
        headless = get_env_bool("BROWSER_HEADLESS", True)
        disable_security = get_env_bool("BROWSER_DISABLE_SECURITY", False)
        chrome_instance_path = os.getenv("BROWSER_CHROME_INSTANCE_PATH", None)
        window_w = int(os.getenv("BROWSER_WINDOW_WIDTH", "1280"))
        window_h = int(os.getenv("BROWSER_WINDOW_HEIGHT", "720"))
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]

        logging.info("Initializing Browser instance...")
        browser = Browser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                chrome_instance_path=chrome_instance_path,
                extra_chromium_args=extra_chromium_args,
            )
        )
        # Yield the browser instance to be available in the context
        yield {"browser": browser}
    except Exception as e:
        logging.error(f"Failed to initialize browser: {e}\n{traceback.format_exc()}")
        # Yield an empty context or re-raise if browser is critical
        yield {}
    finally:
        if browser:
            logging.info("Closing Browser instance...")
            try:
                await browser.close()
                logging.info("Browser instance closed successfully.")
            except Exception as e:
                logging.error(f"Error closing browser: {e}\n{traceback.format_exc()}")

# Initialize FastMCP server with the lifespan manager
app = FastMCP("mcp_server_browser_use", lifespan=server_lifespan)

@app.tool()
async def run_browser_agent(ctx: Context, task: str, memory_name: str | None = None) -> str:
    """
    Runs a browser agent to perform a given task.

    Args:
        ctx: The FastMCP context, providing access to lifespan resources.
        task: The task description for the agent to execute.
        memory_name: Optional name for the memory collection to use.
    """
    browser = ctx.request_context.lifespan_context.get("browser")
    if not browser:
        return "Error: Browser instance is not available. Check server logs."

    try:
        # Get agent and context configuration from environment variables for this specific run
        # Agent Config
        model_provider = os.getenv("MCP_MODEL_PROVIDER", "anthropic")
        model_name = os.getenv("MCP_MODEL_NAME", "claude-3-5-sonnet-20241022") # Ensure this is a valid model
        temperature = float(os.getenv("MCP_TEMPERATURE", "0.7"))
        max_steps = int(os.getenv("MCP_MAX_STEPS", "100"))
        use_vision = get_env_bool("MCP_USE_VISION", True)
        max_actions_per_step = int(os.getenv("MCP_MAX_ACTIONS_PER_STEP", "5"))
        tool_calling_method = os.getenv("MCP_TOOL_CALLING_METHOD", "auto") # or "required" or "none"

        # Browser Context Config (can be dynamic per request if needed)
        window_w = int(os.getenv("BROWSER_WINDOW_WIDTH", "1280"))
        window_h = int(os.getenv("BROWSER_WINDOW_HEIGHT", "720"))
        trace_path = os.getenv("BROWSER_TRACE_PATH") # e.g., "traces/"
        save_recording_path = os.getenv("BROWSER_RECORDING_PATH") # e.g., "recordings/recording.mp4"

        # Memory Config
        memory_enabled = get_env_bool("MEMORY_ENABLED", False)
        memory_config = None
        if memory_enabled:
            default_memory_name = os.getenv("MEMORY_DEFAULT_COLLECTION_NAME", "mem0")
            effective_memory_name = memory_name if memory_name else default_memory_name
            memory_config = {
                "vector_store": {
                    "provider": os.getenv("MEMORY_VECTOR_STORE_PROVIDER", "redis"),
                    "config": {
                        "collection_name": effective_memory_name,
                        "embedding_model_dims": int(os.getenv("MEMORY_EMBEDDING_MODEL_DIMS", 3072)),
                        "redis_url": os.getenv("MEMORY_REDIS_URL", "redis://localhost:6379"),
                    }
                },
                "embedder": {
                    "provider": os.getenv("MEMORY_EMBEDDER_PROVIDER", "vertexai"),
                    "config": {
                        "model": os.getenv("MEMORY_EMBEDDER_MODEL", "text-embedding-large-exp-03-07"),
                        "memory_add_embedding_type": "RETRIEVAL_DOCUMENT",
                        "memory_update_embedding_type": "RETRIEVAL_DOCUMENT",
                        "memory_search_embedding_type": "RETRIEVAL_QUERY"
                    }
                },
                "llm": {
                    "provider": os.getenv("MEMORY_LLM_PROVIDER", model_provider),
                    "config": {
                        "model": os.getenv("MEMORY_LLM_MODEL", model_name),
                    }
                }
            }
            logging.info(f"Memory enabled with collection name: {effective_memory_name}")
        else:
             logging.info("Memory is disabled.")


        # Prepare LLM
        llm = utils.get_llm_model(
            provider=model_provider, model_name=model_name, temperature=temperature
        )

        # Create a new browser context for this task
        logging.info(f"Creating new browser context for task: {task[:50]}...")
        async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path=trace_path,
                save_recording_path=save_recording_path,
                no_viewport=False, # Usually want a viewport for realistic interaction
                browser_window_size=BrowserContextWindowSize(
                    width=window_w, height=window_h
                ),
            )
        ) as browser_context:
            logging.info("Browser context created. Initializing Agent...")
            # Create controller and agent for this specific task run
            controller = Controller() # Use default controller unless custom functions needed
            agent = Agent(
                task=task,
                use_vision=use_vision,
                llm=llm,
                browser=browser, # Pass the shared browser instance
                browser_context=browser_context, # Pass the task-specific context
                controller=controller,
                max_actions_per_step=max_actions_per_step,
                # injected_agent_state is removed - state is managed per run
                tool_calling_method=tool_calling_method,
                enable_memory=memory_enabled,
                memory_config=memory_config,
            )

            logging.info(f"Running agent for task: {task[:50]}...")
            # Run agent with improved error handling
            try:
                with redirect_stdout(sys.stderr):
                    history = await agent.run(max_steps=max_steps)
                final_result = (
                    history.final_result()
                    or f"Task completed, but no specific final result extracted. Full history: {history}"
                )
                logging.info(f"Agent finished successfully for task: {task[:50]}. Result: {final_result[:100]}...")
                return final_result
            except asyncio.CancelledError:
                logging.warning(f"Task was cancelled: {task[:50]}")
                return "Task was cancelled"
            except Exception as e:
                logging.error(f"Agent run error for task '{task[:50]}': {str(e)}\n{traceback.format_exc()}")
                # Provide a more informative error message back to the MCP client
                return f"Error during task execution: {type(e).__name__} - {str(e)}. Check server logs for details."
            # Context is automatically closed by 'async with'

    except Exception as e:
        # Catch errors during setup (before agent.run)
        logging.error(f"Error setting up or running browser agent for task '{task[:50]}': {str(e)}\n{traceback.format_exc()}")
        return f"Error setting up task: {type(e).__name__} - {str(e)}. Check server logs for details."
    # No manual cleanup needed here, lifespan and async with handle it

def main():
    # Run the FastMCP application using stdio transport by default
    # The lifespan manager ensures the browser is handled correctly
    app.run(transport='stdio')

if __name__ == "__main__":
    main()
