from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.client.base import ConversationDict
from langchain_google_genai import ChatGoogleGenerativeAI

import chainlit as cl


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model  = ChatOpenAI(streaming=True, model="gpt-3.5-turbo", temperature=0.5, max_retries=5)
    # model = ChatGoogleGenerativeAI(streaming=True, model="gemini-pro", temperature=1)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an enthusiastic computer software and hardware expert that is here to help people learn about somputers in general. Assume that questions asked are from people who have little knowledge of computers. Provide snippets and learning resources for each of your answers. Be enthusiatic in your responses and DON'T FORGET TO PROVIDE EXAMPLES!!!!"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


# @cl.password_auth_callback
# def auth():
#     return cl.AppUser(username="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()


@cl.on_chat_resume
async def on_chat_resume(conversation: ConversationDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in conversation["messages"] if m["parentId"] == None]
    for message in root_messages:
        if message["authorIsUser"]:
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)