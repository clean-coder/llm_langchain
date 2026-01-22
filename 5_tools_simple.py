from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import tool

"""
1. Tool Definition with @tool decorator:

@tool
def get_forecast(city: str) -> str:
    #Get weather forecast for a specified city.

- The @tool decorator converts a regular function into a LangChain tool
- The docstring is important - the LLM uses it to understand when to call the tool
- Type hints help the LLM understand the parameters

2. Bind tools to the LLM:

llm_with_tools = llm.bind_tools([get_forecast])

- This tells the LLM about available tools
- The LLM can now decide when to call get_forecast


3. Tool execution loop:

while response.tool_calls:
    # Execute tools and add results to history

- Checks if the LLM wants to use any tools
- Executes the tools and adds results as ToolMessage

4. ToolMessage:

ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call["id"]
)

- Special message type for tool results
- Links the result back to the specific tool call

5. Sends conversation history to the LLM:

response = llm_with_tools.invoke(chat_history)

- Gets a final response from the LLM


6. Example Conversation Flow:

USER: "What's the weather like in Paris?"
LLM : thinks: "I should use the get_forecast tool"
USER: executes Tool for the LLM by calling: get_forecast(city="Paris")
      Result: "Temperature: 18°C, Conditions: Partly cloudy..."
      sends ToolMessage with result back to LLM
LLM : gets the result and formulates a natural response
USER: sees: "The weather in Paris is partly cloudy with a temperature of 18°C..."


7. Problem: Warming up the tool usage

- Sometimes the LLM does not call the tool on the first try.
- This is a common issue with LLMs and tool calling - sometimes they need a "warm-up" or they don't reliably use tools on the first call.
- FIX: 
    a) Re-attempt the same question to encourage tool usage.
    b) or create before the first question a system prompt that instructs the model to use the tool when relevant.
"""

# Define the forecast tool
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
    
    forecast = forecasts.get(city, f"Sorry, no forecast available for {city}")
    print(f'   weather forcast for {city}: {forecast}')
    return forecast


# conversation history
chat_history = []
    
def chat_with_tools(llm_with_tools: BaseChatOpenAI, user_message: str) -> str:
    chat_history.append(HumanMessage(content=user_message))
    
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


def _print_line():
    print("-" * 60) 

if __name__ == "__main__":
    model = "llama3.1"
    llm = ChatOllama(model=model, temperature=0)
    llm_with_tools = llm.bind_tools([get_forecast])

    question ='What kind of clothes do I need for a short trip to Paris?'
    print(f"\nUser: {question}")

    _print_line()
    print("\nFirst attempt:")
    response1 = chat_with_tools(llm_with_tools, question)
    print(f"\nAI: {response1}\n")

    _print_line()
    print("\nSecond attempt (warming up the tool usage):")
    response2 = chat_with_tools(llm_with_tools, question)
    print(f"\nAI: {response2}\n")
