from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
    

# define output schema
class LibraryOutput(BaseModel):
    name: str = Field(description="name of a library")
    provider: str = Field(description="provider of the library")
    url: str = Field(description="URL of the library")
    language: str = Field(description="programming language of the library")
    version: str = Field(description="version of the library")


class LibrariesOutput(BaseModel):
    libraries: list[LibraryOutput] = Field(description="list of libraries")


def print_result(result: LibrariesOutput) -> None:
    print(result)
    libraries = result.model_dump()["libraries"]
    for library in libraries:
        print(f"Name: {library['name']}")
        print(f"Provider: {library['provider']}")
        print(f"Language: {library['language']}")
        print(f"Version: {library['version']}")
        print(f"Url: {library['url']}")
        print("- "*30)

# define prompt template with format instructions (to be filled in by the parser)
messages = [
    ("system", "What are the most popular programming libraries for accessing LLMs?  Please use the following schema {format_instructions}"),
    ("user", "Programming Language: {programming_language}, Number of Libraries: {library_count}")
]

# output parser
parser = PydanticOutputParser(pydantic_object=LibrariesOutput)

prompt_template = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())

MODEL_NAME = "openai/gpt-oss-120b"
model = ChatGroq(model=MODEL_NAME)

# define the chain
chain = prompt_template | model | parser

inputs = {
    "programming_language": "Python",
    "library_count": 5
}
result: LibrariesOutput = chain.invoke(input=inputs)
print_result(result)   