import json


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
