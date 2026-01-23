from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolCall
import streamlit as st

# chatbot with memory (using st.session_state for chat history) and tool support
#
# run with: 
#   source .venv/bin/activate 
#   streamlit run x_chatbot_ollama_with_streamlit.py
#
# TIP:
# - using the flag USE_TOOLS (in the source code) you can enable/disable tool usage
# - all kind of messages (HumanMessage, AIMessage, SystemMessage, ToolMessage) are supported. Filter them manuall in display_chat_history()

class Conversation:
    def __init__(self, llm: BaseChatModel, use_tools: bool = True):
        self._init_chat_history()
        if use_tools:
            self.llm = llm.bind_tools([get_forecast]) # llm with tools
            self._set_system_prompt()
        else:
            self.llm = llm  # plain llm without tools

    def _init_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def ask(self, user_message: str):        
        # add user message
        st.session_state.messages.append(HumanMessage(content=user_message))
        
        # get response from LLM
        response = self.llm.invoke(st.session_state.messages)
        st.session_state.messages.append(response)

        # check if the model wants to use a tool
        while response.tool_calls:
            # execute each tool call
            for tool_call in response.tool_calls:
                self._make_tool_call_and_add_result_to_history(tool_call)
            
            # get final response from LLM with tool results
            response = self.llm.invoke(st.session_state.messages)
            st.session_state.messages.append(response)

        return response.content


    def _make_tool_call_and_add_result_to_history(self, tool_call: ToolCall):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # execute the tool
        if tool_name == "get_forecast":
            tool_result = get_forecast.invoke(tool_args)
        
            # add tool result to history
            st.session_state.messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )
        
    def _set_system_prompt(self):
        SYSTEM_PROMPT = """You are a helpful travel assistant.

You have access to a get_forecast tool that provides weather information for cities.

ONLY use the get_forecast tool when the user's question is directly about:
- Weather conditions in a specific city
- Temperature in a specific city
- What to pack or wear for a trip to a specific city
- Planning activities based on weather

For all other questions (math, general knowledge, coding, etc.), answer directly WITHOUT using any tools.
"""

        # only set system prompt if messages is empty or doesn't have a system message
        if not st.session_state.messages or not isinstance(st.session_state.messages[0], SystemMessage):
            st.session_state.messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    
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


def initial_ui(use_tools: bool):
    if use_tools:
        st.title("🌤️  My Chatbot")
        st.write("Ask me about what to pack for a trip to a city!")
    else:
        st.title("🤖  My Chatbot")
        st.write("Ask me anything!")


def display_chat_history():
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            with st.chat_message("assistant", avatar="⚙️"):
                st.markdown(f"**System Prompt:**\n\n{message.content}")
        elif isinstance(message, ToolMessage):
            with st.chat_message("assistant", avatar="🔧"):
                st.markdown(f"**Tool Result:**\n\n{message.content}")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # show tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    st.markdown("🔧 **Using tool:**")
                    for tool_call in message.tool_calls:
                        st.code(f"{tool_call['name']}({tool_call['args']})", language="python")
                # show the actual response content if it exists
                if message.content:
                    st.markdown(message.content)


def run_conversation(conversation: Conversation, use_tools: bool):
    prompt = _create_prompt_input(use_tools)
    if not prompt:
        return
    
    conversation.ask(prompt)
    st.rerun() # trigger a rerun to display the updated history


def _create_prompt_input(use_tools: bool):
    if use_tools:
        return st.chat_input("Ask me about what to pack for a trip...")    
    else:
        return st.chat_input("Ask me anything...")


if __name__ == "__main__":
    USE_TOOLS = True
    initial_ui(USE_TOOLS)

    conv = Conversation(ChatOllama(model="llama3.1"), use_tools=USE_TOOLS)
 
    display_chat_history()
    run_conversation(conv, use_tools=USE_TOOLS)