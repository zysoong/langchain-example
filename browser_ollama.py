from typing import Sequence

from langchain.agents.output_parsers.ollama_tools import OllamaToolsAgentOutputParser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_experimental.llms.ollama_functions import ChatOllama


def create_ollama_tools_agent(
        llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses Ollama tools.
    The prompts will be injected to the tools
    automatically.
    Check the documentation of 'create_openai_tools_agent'
    for detailed instructions.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to. Prompts will be injected as 'prompt'
            attribute automatically.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | prompt
            | llm_with_tools
            | OllamaToolsAgentOutputParser()
    )
    return agent


if __name__ == '__main__':
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user",
         "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
         "and find the previous quests recursively. "),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    sync_browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOllama(model="llama3.1")

    agent = create_ollama_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)
