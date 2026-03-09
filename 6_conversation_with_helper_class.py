from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


"""
Why you need both question AND answer:

- The LLM has no memory between calls
- Each llm.invoke() is independent
- You must send the complete conversation history every time
- If you only stored questions, the LLM wouldn't know what it said before!


What's happening at each step:
After Question 1:

chat_history = [
    HumanMessage("What is the capital of France?"),
    AIMessage("Paris")
]


After Question 2:

chat_history = [
    HumanMessage("What is the capital of France?"),
    AIMessage("Paris"),
    HumanMessage("And Sweden?"),
    AIMessage("Stockholm")
]

After Question 3:

chat_history = [
    HumanMessage("What is the capital of France?"),
    AIMessage("Paris"),
    HumanMessage("And Sweden?"),
    AIMessage("Stockholm"),
    HumanMessage("Which one is larger?"),
    AIMessage("Sweden is larger than France...")
]
"""
MODEL = "llama3.1"
class Conversation:
    def __init__(self):
        self.history = []
    
    def chat(self, question: str):
        llm = ChatOllama(model=MODEL)
        self.history.append(HumanMessage(content=question))
        response = llm.invoke(self.history)
        self.history.append(AIMessage(content=response.content))
        return response.content

if __name__ == "__main__":
    conv = Conversation()

    print("---- First Question ----")
    print(conv.chat("What is the capital of France?"))

    print("\n\n---- Second Question ----")
    print(conv.chat("And Sweden?"))

    print("\n\n---- Third Question ----")
    print(conv.chat("Which one is larger?"))