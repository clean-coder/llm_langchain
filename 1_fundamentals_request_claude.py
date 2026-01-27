from langchain_anthropic import ChatAnthropic
import tools

MODEL = "claude-sonnet-4-5-20250929"
chat_model = ChatAnthropic(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True, # set to False for production
)

response = chat_model.invoke("What is LangChain? Please in max 1 sentences.")
print("---- Raw Output ----")
print(type(response))
print(response.content)

print("\n---- JSON Output ----")
print(tools.prettyfy_json(response.model_dump_json()))
