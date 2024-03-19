import re
import sys
from abc import ABC
from typing import List, Union

from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentOutputParser, LLMSingleActionAgent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langserve import add_routes
import os

from langchain_community.document_loaders import DirectoryLoader
sys.path.append("..")
from settings import XHQB_LOCAL_FILE_PATH


def add_llm_module(app: FastAPI) -> None:

    # 在环境变量里获取OPENAI的key和代理
    os.environ["OPENAI_API_KEY"] = os.environ["LJH_OPENAI_KEY"]
    os.environ["OPENAI_API_BASE"] = os.environ["LJH_OPENAI_BASE"]

    loader = DirectoryLoader(f'{XHQB_LOCAL_FILE_PATH}', glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # 创建工具
    retriever_tool = create_retriever_tool(
        retriever,
        "xhqb_sqlcoder_search",
        "所有sql相关问题都使用这个工具进行回答",
    )
    tools = [retriever_tool]

    # 创建问题和回答的提示词模板
    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Question: {input}
    {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()

    # 初始化默认使用 'gpt-3.5-turbo'
    llm = ChatOpenAI(temperature=0)

    # 包含LLM和提示词的LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    add_routes(
        app,
        agent_executor.with_types(input_type=Input, output_type=Output),
        path="/agent",
    )


# 自定义模板解析类
class CustomPromptTemplate(BaseChatPromptTemplate, ABC):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> List[BaseMessage]:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts

        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


# 自定义输入解析类
class CustomOutputParser(AgentOutputParser):
    @property
    def _type(self) -> str:
        return ""

    def get_format_instructions(self) -> str:
        pass

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # 检查是否得到了最终答案
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str
