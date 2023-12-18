from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper, TextRequestsWrapper, PythonREPL, StackExchangeAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI


import chainlit as cl

@cl.on_chat_start
def start():
    search = GoogleSearchAPIWrapper()
    requests = TextRequestsWrapper()
    python_repl = PythonREPL()
    stackexchange = StackExchangeAPIWrapper()
    yt_search = YouTubeSearchTool()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you need to search things up in general or need to look up software or coding references/examples",
        ),
        Tool(
            name="Requests",
            func=requests.get,
            description="Useful for when you need to make http requests to external URLs",
        ),
        # Tool(
        #     name="python_repl",
        #     description="Usefule for when you need to run pytonn code or repl.",
        #     func=python_repl.run,
        # ),
        Tool(
            name="Stack Exchange",
            description="Usefule for when you want to search technical-based information. You can also use this when finding information about software development.",
            func=stackexchange.run,
        ),
        # Tool(
        #     name="Youtube search",
        #     description="Usefule for when you want to search youtube.",
        #     func=yt_search.run,
        # ),
    ]

    prefix = """You are an enthusiastic computer software and coding expert that helps users solve problems and answer questions. 
    Use the tone of a software engineer tutor and provide thorough explanations for any questions you answer with examples!!
    Respond with written/generated code snippets along with explanations of each snippet. 
    Also provide references and APIs for your responses if necessary.
    Be enthusiatic in your responses. If you do not know the answer, try searching for it. DON'T FORGET TO PROVIDE CODE EXAMPLES!!!
    You have access to the following tools:"""
    # prefix = """Have a conversation with the user
    # You have access to the following tools:"""
    suffix = """Begin!"


    {chat_history}
    
    User Input: {input}
    
    {agent_scratchpad}
    """

    FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
    '''
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    '''

    When you have gathered enough information, write it out to the user.

    '''
    Thought: Do I need to use a tool? No
    AI: [respond]
    '''
    """

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)
    # llm  = ChatOpenAI(streaming=True, model="gpt-3.5-turbo-16k", temperature=0, max_retries=3)
    # llm  = ChatOpenAI(streaming=True, model="gpt-4", temperature=0.5)


    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )


    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True,
        agent_kwargs={
            'prefix': prefix, 
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': suffix
        },
    )

    cl.user_session.set("agent_chain", agent_chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("agent_chain")  # type: AgentExecutor
    # print(f"chain: {str(chain)}")

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    res = await chain.ainvoke(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cb]),
    )

    print(f"res: {res}")