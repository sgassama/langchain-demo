from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True, model="gpt-3.5-turbo", temperature=1)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an enthusiastic computer software and hardware expert that is here to help people learn about somputers in general. Assume that questions asked are from people who have little knowledge of computers. Provide examples and learning resources for each of your answers. Be enthusiatic in your responses.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
