from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import tools

# optional: load environment variables from a .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
    
# ask 3 different LLMs for their favorite libraries for accessing LLMs
# and then consolidate the responses into a single md table

# define output schema
class LibraryOutput(BaseModel):
    name: str = Field(description="name of a library")
    provider: str = Field(description="provider of the library")
    url: str = Field(description="URL of the library")
    language: str = Field(description="programming language of the library")
    version: str = Field(description="version of the library")


class LibrariesOutput(BaseModel):
    libraries: list[LibraryOutput] = Field(description="list of libraries")


def to_string(llm_name: str, result: LibrariesOutput) -> str:
    output_lines = []
    output_lines.append("-" * 60)
    output_lines.append(f"Results from {llm_name}:")
    output_lines.append("-" * 60)

    libraries = result.model_dump()["libraries"]
    for library in libraries:
        output_lines.append(f"Name: {library['name']}")
        output_lines.append(f"Provider: {library['provider']}")
        output_lines.append(f"Language: {library['language']}")
        output_lines.append(f"Version: {library['version']}")
        output_lines.append(f"Url: {library['url']}")
        output_lines.append("- " * 30)
    
    return "\n".join(output_lines)        

PROGRAMMING_LANGUAGE = "Python"
NUMBER_OF_LIBRARIES = 5

# define prompt template with format instructions (to be filled in by the parser)
messages = [
    ("system", "What are the most popular programming libraries for accessing LLMs?  Please use the following schema {format_instructions}"),
    ("user", "Programming Language: {programming_language}, Number of Libraries: {library_count}")
]

# output parser for the defined schema LibrariesOutput
parser = PydanticOutputParser(pydantic_object=LibrariesOutput)

prompt_template = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())

MODEL_CLAUDE = "claude-sonnet-4-5-20250929"
llm_claude = ChatAnthropic(
    model_name=MODEL_CLAUDE,     
    temperature=0.3,        
    verbose=True 
)

MODEL_OPENAI = "gpt-4o-mini"
llm_openai = ChatOpenAI(
    model_name=MODEL_OPENAI,     
    temperature=0.3,        
    verbose=True
)

MODEL_GOOGLE = "gemini-2.5-flash-lite"
chat_model = ChatGoogleGenerativeAI(
    model=MODEL_GOOGLE,  
    temperature=0.3,        
    verbose=True
)

# 1 - create requests for each LLM

# define the chains for each LLM
claude_chain = prompt_template | llm_claude | parser
openai_chain = prompt_template | llm_openai | parser
google_chain = prompt_template | chat_model | parser

# define the parallel chain with named branches claude, openai, and google (which will be the keys in the output dict)
map_chain = RunnableParallel(
    claude=claude_chain, 
    openai=openai_chain,
    google=google_chain
)

inputs = {
    "programming_language": PROGRAMMING_LANGUAGE,
    "library_count": NUMBER_OF_LIBRARIES
}
result: dict[str, LibrariesOutput] = map_chain.invoke(input=inputs)

result_as_string = ""
for llm_name, llm_result in result.items():
    # print(to_string(llm_name, llm_result))
    result_as_string += to_string(llm_name, llm_result) + "\n"


# 2 - consolidate responses into a single table (making a single request to Claude)

template_consolidate_responses = """
The following data represents the responses of several LLMs to the question: 
What are the most popular libraries in {programming_language} for accessing LLMs? 

Can you summarize these providers' responses in one table, with a column for each LLM provider?.
{responses}
"""

prompt_consolidate_responses = ChatPromptTemplate.from_template(template_consolidate_responses)
output_parser_consolidate_responses = StrOutputParser()
chain_consolidate_responses = prompt_consolidate_responses | llm_claude | output_parser_consolidate_responses


response_consolidated = chain_consolidate_responses.invoke({
    "programming_language": PROGRAMMING_LANGUAGE,
    "responses": result_as_string
})  

# print("\n---- Consolidated Response ----")
# print(response_consolidated)

tools.write_data_to_file(response_consolidated, f"libraries_{PROGRAMMING_LANGUAGE.lower()}.md")