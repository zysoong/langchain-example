from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser
)
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from prompt import *

if __name__ == '__main__':
    prompt = ffxiv_previous_request_prompt_openai

    sync_browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)
