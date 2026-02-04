import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tools

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY not set")
else:
    print(f'GROQ_API_KEY loaded successfully: {os.getenv("GROQ_API_KEY")[:10]}...\n')

# list of available models: https://console.groq.com/docs/models
MODEL="openai/gpt-oss-120b"
chat_model = ChatGroq(
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

