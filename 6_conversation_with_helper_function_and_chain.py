from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

MODEL = "llama3.1"
chat_history = []

def chat(question: str, chat_history: list[BaseMessage]) -> str:
    chain = _create_chain()
    response = chain.invoke({
        "chat_history": chat_history,
        "question": question,
    })

    # Update history after each turn
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

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
    print(chat("What is the capital of France?", chat_history))

    print("\n\n---- Second Question ----")
    print(chat("And Sweden?", chat_history))

    print("\n\n---- Third Question ----")
    print(chat("Which one is larger?", chat_history))
