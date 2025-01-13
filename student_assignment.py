import requests
import base64
from mimetypes import guess_type

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    response = agent_with_chat_history.invoke({"input":question}, config={"configurable": {"session_id": "<foo>"}}).get('output')
    
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
    generate_hw02(question2)
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
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response_schemas = [
        ResponseSchema(
            name="add",
            description="該紀念日是否需要加入先前的清單內,若月份相同且該紀念日不被包含在清單內,則回true,否則為false,只能是這兩種答案"),
        ResponseSchema(
            name="reason",
            description="決定該紀念日是否加入清單的理由")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions}"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = agent_with_chat_history.invoke({"input":prompt.format(question=question3)}, config={"configurable": {"session_id": "<foo>"}}).get('output')
    response = result(llm, response)
    return response
    
def generate_hw04(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    image_url = local_image_to_data_url()
    response_schemas = [
        ResponseSchema(
            name="score",
            description="圖片文字表格中顯示的指定隊伍的積分數")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "辨識圖片中的文字表格,{format_instructions}"),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                ],
            ),
            ("human", "{question}")
        ]
    )
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format(question=question)).content
    response = result(llm, response)
    return response
    
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

def local_image_to_data_url():
    # Example usage
    image_path = 'baseball.png'
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

question3 = "根據先前的節日清單，這個節日{'date': '10-31', 'name': '蔣公誕辰紀念日'}是否有在該月份清單？"
print(generate_hw04('請問中華台北的積分是多少'))
