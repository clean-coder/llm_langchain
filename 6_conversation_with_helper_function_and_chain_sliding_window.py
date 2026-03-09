from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

MODEL = "llama3.1"
chat_history = []

MAX_HISTORY = 4  # keep last 4 messages (2 turns)

def chat_with_sliding_window(question: str, chat_history: list[BaseMessage], debug: bool = True) -> str:
    chain = _create_chain()

    response = chain.invoke({
        "chat_history": chat_history[-MAX_HISTORY:],  # sliding window
        "question": question,
    })

    if debug:
        _print_history(chat_history[-MAX_HISTORY:])  # debug print current window

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

def _print_history(history: list[BaseMessage]):
    print("  DEBUG: Current history:")
    if not history:
        print("  - (empty)")
    else:
        for msg in history:
            print(f"  - {msg.__class__.__name__}: {msg.content}")
    print("  ---")

def _create_chain():
    # Key: use MessagesPlaceholder to inject history into the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # history slot
        ("human", "{question}"),                            # current question
    ])

    llm = ChatOllama(model=MODEL)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


if __name__ == "__main__":
    print("---- First Question ----")
    print(chat_with_sliding_window("What is the capital of France?", chat_history))

    print("\n\n---- Second Question ----")
    print(chat_with_sliding_window("And Sweden?", chat_history))

    print("\n\n---- Third Question ----")
    print(chat_with_sliding_window("And Germany?", chat_history))

    print("\n\n---- Fourth Question ----")
    print(chat_with_sliding_window("And Switzerland?", chat_history))

    print("\n\n---- Fifth Question (triggers sliding window) ----")
    print("Dummy questions to push out old history...")
    chat_with_sliding_window("Lorem ipsum?", chat_history, debug=False) 
    chat_with_sliding_window("Lorem ipsum?", chat_history, debug=False)
    chat_with_sliding_window("Lorem ipsum?", chat_history, debug=False)
    chat_with_sliding_window("Lorem ipsum?", chat_history, debug=False)
    
    print("\n\n---- Sixth Question (check history after sliding) ----")
    print("history does no longer contains any useful infromation")
    print(chat_with_sliding_window("And Switzerland?", chat_history, debug=False))

