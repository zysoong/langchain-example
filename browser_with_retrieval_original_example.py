from langchain_wire.agent.openai_tools import create_openai_tools_agent_and_inject_prompts
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser
)
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_wire.toolkit.toolkit import RetrievalPlayWrightBrowserToolkit
from prompt import *

if __name__ == '__main__':
    prompt = ffxiv_previous_request_prompt_openai

    sync_browser = create_sync_playwright_browser()
    toolkit = RetrievalPlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = create_openai_tools_agent_and_inject_prompts(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)
