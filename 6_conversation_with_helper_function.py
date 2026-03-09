from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

chat_history = []

def chat(user_message: str, chat_history: list[BaseMessage]) -> str:
    llm = ChatOllama(model="llama3.1")
    chat_history.append(HumanMessage(content=user_message))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    return response.content

if __name__ == "__main__":
    print("---- First Question ----")
    print(chat("What is the capital of France?", chat_history))

    print("\n\n---- Second Question ----")
    print(chat("And Sweden?", chat_history))

    print("\n\n---- Third Question ----")
    print(chat("Which one is larger?", chat_history))