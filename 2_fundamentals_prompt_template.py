from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

MODEL = "gpt-4o-mini"
chat_model = ChatOpenAI(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True,  
)

template = "Give me a brief summary about the color {color}."

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | chat_model
print(f'Type of Chain: {type(chain)}')

response_red = chain.invoke({"color":"red"})
print(response_red.content)

response_blue = chain.invoke({"color":"blue"})
print(response_blue.content)
