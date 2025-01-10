import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    response_schemas = [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions},有幾個答案就回答幾次,將所有答案放進同個list"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format(question=question)).content
    response = result(llm, response)
    return response

def get_holiday(year: int, month: int) -> str:
    """ Check the holidays for specific year and month
    Args:
        year: year
        month: month
    """
    url = f"https://calendarific.com/api/v2/holidays?&api_key=NWLS9VUCWzUG0H0jpGNem9yIR59dbftW&country=tw&year={year}&month={month}"
    response = requests.get(url)
    data = response.json()
    return data.get('response')

class GetHoliday(BaseModel):
    """ Get holidays"""

    year: int = Field(..., description="specific year")
    month: int = Field(..., description="specific month")

def generate_hw02(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    instructions = "You are an agent."
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    tool = StructuredTool.from_function(
        name="get_holiday",
        description="Fetch holidays for specific year and month",
        func=get_holiday,
        args_schema=GetHoliday,
    )
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    response = agent_executor.invoke({"input":question}).get('output')
    
    response_schemas = [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將我提供的資料整理成指定格式,使用台灣語言並回答問題,{format_instructions},有幾個答案就回答幾次,將所有答案放進同個list"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format(question=response)).content
    response = result(llm, response)
    return response
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

def result(llm, data):
    response_schemas = [
        ResponseSchema(
            name="Result",
            description="json內的所有內容",
            type="list")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將提供的json內容輸出成指定json格式,{format_instructions}"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format(question=data)).content

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "將我提供的文字進行處理，若第一行內容為'```json'，將第一行移除"),
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(prompt.invoke(input = response)).content
    return response

print(generate_hw01("2024年台灣10月紀念日有哪些?"))
