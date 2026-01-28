import json
from langchain_core.messages.ai import AIMessage

def prettyfy_json(raw_json: str) -> str:
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


def print_token_usage(response: AIMessage) -> None:
    print("\n---- Token Usage ----")
    print(f"Prompt Tokens: {response.response_metadata['token_usage']['prompt_tokens']}")
    print(f"Completion Tokens: {response.response_metadata['token_usage']['completion_tokens']}")
    print(f"Total Tokens: {response.response_metadata['token_usage']['total_tokens']}")
    print()