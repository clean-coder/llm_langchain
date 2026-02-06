from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

MODEL = "gpt-4o-mini"
chat_model = ChatOpenAI(
    model_name=MODEL,     
    temperature=0.3,        
    verbose=True,    
)

template = """
Give me the first {number} prime numbers and sum of them.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = JsonOutputParser()

chain = prompt | chat_model | output_parser

response_as_json = chain.invoke({
    "number":5,
    "format_instructions": output_parser.get_format_instructions()
    })

print("\n---- Output as JSON ----")
print(f'Type of response: {type(response_as_json)}')
print(response_as_json)