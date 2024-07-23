def get_request_list(instructions, sampling_params={}):
    if isinstance(instructions, list):
        request_list = [
            {"messages": [{"role": "user", "content": prompt}], **sampling_params} for prompt in instructions
                    ]
    elif isinstance(instructions, str):
        request_list = [{"messages": [{"role": "user", "content": instructions}], **sampling_params}]
    else:
        raise TypeError("Instructions must be either a list of strings or a single string.")
    return request_list