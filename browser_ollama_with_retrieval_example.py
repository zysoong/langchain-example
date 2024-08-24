from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langchain_wire.agent.ollama_tools import create_ollama_tools_agent_and_inject_prompts
from langchain_wire.toolkit.toolkit import RetrievalPlayWrightBrowserToolkit

if __name__ == '__main__':
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user",
         "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
         "and find the previous quests recursively. "),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    sync_browser = create_sync_playwright_browser()
    toolkit = RetrievalPlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOllama(model="mistral-nemo", temperature=0.0)

    agent = create_ollama_tools_agent_and_inject_prompts(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)
