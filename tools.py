import json
from langchain_core.messages.ai import AIMessage


def prettyfy_json(response: AIMessage) -> str:
    # get raw json string from response
    raw_json = _response_to_json(response)

    # load json string into a Python object
    data = json.loads(raw_json)

    # dump it back out, *pretty‑printed*
    return json.dumps(
        data,
        indent=4,                 # 4‑space indentation (default is 2 in Python 3.10+)
        sort_keys=True,           # optional: sort the keys alphabetically
        ensure_ascii=False,       # keep Unicode characters instead of \uXXXX
        separators=(', ', ': ')   # pretty separators (comma+space, colon+space)
    )    

def _response_to_json(response: AIMessage) -> str:
    return response.model_dump_json()


def print_token_usage(response: AIMessage) -> None:
    print(f"Prompt Tokens    : {response.usage_metadata['input_tokens']}")
    print(f"Completion Tokens: {response.usage_metadata['output_tokens']}")
    print(f"Total Tokens     : {response.usage_metadata['total_tokens']}")        
    print()


def write_data_to_file(data: str, filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)