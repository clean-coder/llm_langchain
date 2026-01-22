from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import tool

@tool
def get_forecast(city: str) -> str:
    """Get weather forecast for a specified city.
    
    Args:
        city: The name of the city to get the forecast for
        
    Returns:
        Weather forecast information as a string
    """
    forecasts = {
        "Paris": "Temperature: 18°C, Conditions: Partly cloudy, Wind: 10 km/h",
        "Stockholm": "Temperature: 12°C, Conditions: Rainy, Wind: 15 km/h",
        "London": "Temperature: 15°C, Conditions: Foggy, Wind: 8 km/h",
        "Berlin": "Temperature: 16°C, Conditions: Sunny, Wind: 12 km/h",
        "Madrid": "Temperature: 24°C, Conditions: Clear skies, Wind: 5 km/h"
    }
    
    return forecasts.get(city, f"Sorry, no forecast available for {city}")


# System prompt that encourages tool usage
SYSTEM_PROMPT = """You are a helpful travel assistant. 

IMPORTANT: When users ask about:
- What to pack for a trip
- What clothes to bring
- Weather conditions
- Temperature in a city

You MUST use the get_forecast tool to check the current weather before providing advice. Never give generic packing advice without checking the actual weather forecast first."""


def chat_with_tools(llm_with_tools: BaseChatOpenAI, user_message: str, show_message_history: bool) -> str:   
    # add system prompt at the beginning of the conversation 
    chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

    # add user message
    chat_history.append(HumanMessage(content=user_message))
    
    # get response from LLM
    response = llm_with_tools.invoke(chat_history)
    chat_history.append(response)
    
    # check if the model wants to use a tool
    while response.tool_calls:
        # execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n🔧 Calling tool: {tool_name} with args: {tool_args}")
            
            # execute the tool
            if tool_name == "get_forecast":
                tool_result = get_forecast.invoke(tool_args)
                print(f"   weather forecast: {tool_result}")
            
            # add tool result to history
            chat_history.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )
        
        # get final response from LLM with tool results
        response = llm_with_tools.invoke(chat_history)
        chat_history.append(response)

    if show_message_history:
        _print_message_history(chat_history)
    
    # Uncomment to see the full JSON response
    # print(tools.prettyfy_json(response.model_dump_json()))
    return response.content


def _print_message_history(chat_history: list[BaseMessage]):
    print("\n---- Full Message History ----")
    for index, message in enumerate(chat_history):
        print(f"Message {index}\n  {message}\n\n")  
    print("\n---- Full Message History END ----\n\n")


if __name__ == "__main__":
    model = "llama3.1"
    llm = ChatOllama(model=model)
    llm_with_tools = llm.bind_tools([get_forecast])    

    question = 'What kind of clothes do I need for a short trip to Paris?'
    print(f"\nUser: {question}")

    response = chat_with_tools(llm_with_tools, question, show_message_history=False)
    print(f"AI: {response}\n")  
