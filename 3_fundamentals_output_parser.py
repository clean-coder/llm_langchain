from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "gpt-4o-mini"
chat_model = ChatOpenAI(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True,    
)

template = "Give me a brief summary about the color {color}."

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser
print(f'Type of Chain: {type(chain)}')

response_as_string = chain.invoke({"color":"red"})

print("\n---- Raw Output ----")
print(f'Type of response: {type(response_as_string)}')
print(response_as_string)