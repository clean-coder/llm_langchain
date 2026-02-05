import os
from langchain_anthropic import ChatAnthropic
import tools

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

if os.getenv("ANTHROPIC_API_KEY") is None:
    raise ValueError("ANTHROPIC_API_KEY not set")
else:
    print(f'ANTHROPIC_API_KEY loaded successfully: {os.getenv("ANTHROPIC_API_KEY")[:10]}...\n')

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
print(tools.prettyfy_json(response))

print("\n---- Token Usage ----")
tools.print_token_usage(response)     

