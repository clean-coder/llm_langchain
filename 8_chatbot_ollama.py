from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# define the forecast tool
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


# system prompt that encourages tool usage
SYSTEM_PROMPT = """You are a helpful travel assistant. 

IMPORTANT: When users ask about:
- What to pack for a trip
- What clothes to bring
- Weather conditions
- Temperature in a city

You MUST use the get_forecast tool to check the current weather before providing advice. Never give generic packing advice without checking the actual weather forecast first."""


# chat history with system prompt
chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

# chat with tools supported
def ask(user_message):
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
    
    return response.content


def _draw_line() -> None:
    print("=" * 60)     


if __name__ == "__main__":
    MODEL = "llama3.1"
    llm = ChatOllama(model=MODEL)
    llm_with_tools = llm.bind_tools([get_forecast])

    _draw_line()
    question_1 = 'What kind of clothes do I need for a short trip to Paris?'
    print(f"\nUser: {question_1}")
    response1 = ask(question_1)
    print(f"AI: {response1}\n")

    _draw_line()
    question_2 = 'And for Stockholm?'
    print(f"\nUser: {question_2}")
    response2 = ask(question_2)
    print(f"AI: {response2}\n")
