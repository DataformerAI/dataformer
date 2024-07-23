def format_request_list(instructions):
    if isinstance(instructions, list):
        request_list = [
            {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
        ]
    elif isinstance(instructions, str):
        request_list = [{"messages": [{"role": "user", "content": instructions}]}]
    else:
        raise TypeError("Instructions must be either a list of strings or a single string.")
    return request_list