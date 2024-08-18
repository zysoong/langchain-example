from langchain.agents.openai_tools.base import create_openai_tools_agent_and_inject_prompts
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser
)
from langchain.agents import AgentExecutor

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits.playwright.toolkit import RetrievalPlayWrightBrowserToolkit


if __name__ == '__main__':
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. "),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
                  "and find previous quests recursively. "
                  "Response with 5 previous quests of {quest}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

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
