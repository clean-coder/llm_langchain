from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tools

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

response = chain.invoke({"color":"red"})
print("\n---- Raw Output ----")
print(f'Type of response: {type(response)}')
print(response.content)

print("\n---- JSON Output ----")
print(tools.prettyfy_json(response.model_dump_json()))
