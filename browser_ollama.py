from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)

from langchain.agents import AgentExecutor
from langchain_ollama.chat_models import ChatOllama
from langchain_wire.agent.ollama_tools import create_ollama_tools_agent
from prompt import ffxiv_previous_request_prompt

if __name__ == '__main__':
    prompt = ffxiv_previous_request_prompt

    sync_browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOllama(model="mistral-nemo", temperature=0.0)

    agent = create_ollama_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)
