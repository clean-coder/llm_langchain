from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
import streamlit as st

# simple chatbot (using st.session_state for chat history)
#
# run with: 
#   source .venv/bin/activate 
#   streamlit run x_chatbot_ollama_with_streamlit_simple.py

class Conversation:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm  
        self._init_chat_history()

    def ask(self, user_message: str) -> None:        
        self._append_to_chat_history(HumanMessage(content=user_message))
        response = self.llm.invoke(st.session_state.messages)
        self._append_to_chat_history(response)

    def _init_chat_history(self) -> None:
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def _append_to_chat_history(self, message) -> None:
        st.session_state.messages.append(message)


def display_chat_history() -> None:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): # icon for user
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): # icon for assistant
                st.write(message.content)


def start_conversation(conversation: Conversation) -> None:
    prompt = st.chat_input("Ask me about ...")
    if prompt:
        conversation.ask(prompt)
        st.rerun() # to refresh the chat display


if __name__ == "__main__":
    conversation = Conversation(ChatOllama(model="llama3.1"))
    display_chat_history()
    start_conversation(conversation)