from langchain_ollama import ChatOllama
import tools

MODEL = "llama3.1"
chat_model = ChatOllama(model=MODEL)

response = chat_model.invoke("What is LangChain? Please in max 1 sentences.")

print("---- Raw Output ----")
print(type(response))
print(response.content)

print("\n---- JSON Output ----")
print(tools.prettyfy_json(response.model_dump_json())) 
