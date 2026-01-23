from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

chat_history = []

def chat(user_message, chat_history, llm):
    chat_history.append(HumanMessage(content=user_message))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    return response.content

if __name__ == "__main__":
    llm = ChatOllama(model="llama3.1")

    print("---- First Question ----")
    print(chat("What is the capital of France?", chat_history, llm))

    print("\n\n---- Second Question ----")
    print(chat("And Sweden?", chat_history, llm))

    print("\n\n---- Third Question ----")
    print(chat("Which one is larger?", chat_history, llm))