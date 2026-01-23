from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

def chat(question: str) -> str: 
    MODEL = "llama3.1"
    llm = ChatOllama(model=MODEL)

    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template(question)
    chain = prompt | llm | output_parser

    return chain.invoke({})

print("---- First Question ----")
response = chat("What is the capital of France.")
print(response)

print("\n\n---- Second Question ----")
response = chat("And Sweden?")
print(response) 
