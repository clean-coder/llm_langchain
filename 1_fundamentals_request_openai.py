from langchain_openai import ChatOpenAI
import tools

MODEL = "gpt-4o-mini"
chat_model = ChatOpenAI(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True, # set to False for production
)

response = chat_model.invoke("What is LangChain? Please in max 1 sentences.")

print("---- Raw Output ----")
print(type(response))
print(response.content)

print("\n---- JSON Output ----")
print(tools.prettyfy_json(response))

print("\n---- Token Usage ----")
tools.print_token_usage(response)   

