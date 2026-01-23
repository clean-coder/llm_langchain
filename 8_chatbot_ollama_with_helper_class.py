from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolCall

class Conversation:
    def __init__(self, llm: BaseChatModel, use_tools: bool = True, debug: bool = False):
        self.debug = debug
        self.chat_history = []
        if use_tools:
            self.llm = llm.bind_tools([get_forecast]) # llm with tools
            self._set_system_prompt()
        else:
            self.llm = llm  # plain llm without tools
    

    def ask(self, user_message: str):
        # add user message
        self.chat_history.append(HumanMessage(content=user_message))
        
        # get response from LLM
        response = self.llm.invoke(self.chat_history)
        self.chat_history.append(response)
        
        # check if the model wants to use a tool
        while response.tool_calls:
            # execute each tool call
            for tool_call in response.tool_calls:
                self._make_tool_call_and_add_result_to_history(tool_call)
            
            # get final response from LLM with tool results
            response = self.llm.invoke(self.chat_history)
            self.chat_history.append(response)
        
        return response.content


    def _make_tool_call_and_add_result_to_history(self, tool_call: ToolCall):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if self.debug:
            print(f"\n🔧 Calling tool: {tool_name} with args: {tool_args}")
        
        # execute the tool
        if tool_name == "get_forecast":
            tool_result = get_forecast.invoke(tool_args)
            if self.debug:
                print(f"   weather forecast: {tool_result}")
        
        # add tool result to history
        self.chat_history.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )
        
    def _set_system_prompt(self):
        SYSTEM_PROMPT = """You are a helpful travel assistant. 

        IMPORTANT: When users ask about:
        - What to pack for a trip
        - What clothes to bring
        - Weather conditions
        - Temperature in a city

        You MUST use the get_forecast tool to check the current weather before providing advice. Never give generic packing advice without checking the actual weather forecast first."""

        self.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]


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


def chat_without_tools_example():
    model = "llama3.1"
    conv = Conversation(ChatOllama(model=model), use_tools=False)

    print("\n--- Chat without tools ---")
    print(conv.ask("What is the capital of France?"))
    print(conv.ask("And Sweden?"))


def chat_with_tools_example():
    model = "llama3.1"
    conv = Conversation(ChatOllama(model=model), use_tools=True, debug=True)

    print("\n--- Chat with tools ---")

    question = "\n> What kind of clothes do I need for a short trip to Paris?"
    print(question)
    print(f'\n\n{conv.ask(question)}')

    question = "\n\n> And what about Stockholm?"
    print(question)
    print(f'\n\n{conv.ask(question)}')

if __name__ == "__main__":
    chat_without_tools_example()
    chat_with_tools_example()