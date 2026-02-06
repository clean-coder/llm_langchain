from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "gpt-4o-mini"
chat_model = ChatOpenAI(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True,    
)

template = "Give me the first {number} prime numbers."

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser

response_as_string = chain.invoke({"number":5})

print("\n---- Output as String ----")
print(f'Type of response: {type(response_as_string)}')
print(response_as_string)