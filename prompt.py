from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ffxiv_previous_request_prompt_ollama = ChatPromptTemplate.from_messages([
        ("system", "You are requested to perform some browser operations to websites. The operations have been "
                   "altered to yield the result of the browser operations. "
                   "Use the provided tools to answer the question. "
                   "Only give the arguments in your answer which followed the property schema of the provided tool. "
                   ""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user",
         "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
         "and find the previous quests recursively. \n"
         ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
])

ffxiv_previous_request_prompt_ollama_simplified = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user",
         "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
         "and find the previous quests recursively. \n"
         ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
])

ffxiv_previous_request_prompt_openai = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. "),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} "
              "and find previous quests recursively. "
              "Response with 5 previous quests of {quest}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

