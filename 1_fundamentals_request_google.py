from langchain_google_genai import ChatGoogleGenerativeAI
import tools

# list of models: https://ai.google.dev/gemini-api/docs/models

MODEL = "gemini-2.5-flash-lite"
chat_model = ChatGoogleGenerativeAI(
    model=MODEL,  # Note: parameter is 'model' not 'model_name'
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

