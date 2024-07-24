import copy
import json
import logging

def get_request_list(instructions, sampling_params={}, system_prompt=""):
    if isinstance(instructions, list):
        request_list = [
            {"messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + 
                         [{"role": "user", "content": prompt}], 
             **sampling_params} for prompt in instructions
        ]
    elif isinstance(instructions, str):
        request_list = [
            {"messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + 
                         [{"role": "user", "content": instructions}], 
             **sampling_params}
        ]
    else:
        raise TypeError("Instructions must be either a list of strings or a single string.")
    return request_list

def save_jsonl(data, filename="data.jsonl", ensure_ascii=False):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=ensure_ascii)
            f.write('\n')

def add_system_prompt(messages, system_prompt):

    messages = copy.deepcopy(messages)
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = system_prompt
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages

def get_messages(data, system_prompt=""):

    data = copy.deepcopy(data)
    
    messages = []
    
    for data_row in data:
    
        if len(data_row) >= 2 and isinstance(data_row, list):
            if "choices" in data_row[1]:
                # Response List
                final_messages = data_row[0]["messages"] + [{"role": "assistant", "content": data_row[1]["choices"][0]["message"]["content"]}]
        else:
            # Request List
            final_messages = data_row["messages"]

        if system_prompt:
            final_messages = add_system_prompt(final_messages, system_prompt)

        messages.append(final_messages)

    return messages

def get_sharegpt(data, system_prompt="", save_file=None):

    data = copy.deepcopy(data)

    messages = get_messages(data, system_prompt=system_prompt)

    for messages_row in messages:

        # Convert to sharegpt format
        for message in messages_row:  
            if 'role' in message:  # Check if 'role' key exists in the dictionary
                message['from'] = message.pop('role')
            if 'content' in message:  # Check if 'content' key exists in the dictionary
                message['value'] = message.pop('content')
            if message.get('from') == 'user':
                message['from'] = 'human'
            elif message.get('from') == 'assistant':
                message['from'] = 'gpt'
    
    if save_file:
        save_jsonl([{"conversations": m} for m in messages], filename=save_file)

    return messages

def get_alpaca(data, input_list=[], system_prompt="", save_file=None):

    data = copy.deepcopy(data)
    
    messages = get_messages(data, system_prompt)

    if input_list:
        if not(len(input_list) == len(data)):
            raise ValueError("Length of data and input_list must be the same.")

    alpaca_data = []

    is_single_turn = all(sum(1 for m in message if m['role'] != 'system') <= 2 for message in messages)

    if not is_single_turn:
        logging.info("Multi-turn conversations - Therefore returning text instead of instruction, input, output")

    for idx, message in enumerate(messages):
        
        
        if is_single_turn:
            # Single Turn
            alpaca_dict = {}

            if message[0]['role'] == "system":
                alpaca_dict['system'] = message[0]['content']
                alpaca_dict['instruction'] = message[1]['content']
                alpaca_dict['output'] = message[2]['content'] if len(message) >= 3 else ""
            else:
                alpaca_dict['instruction'] = message[0]['content']
                alpaca_dict['output'] = message[1]['content'] if len(message) >= 2 else ""

            if input_list:

                alpaca_dict['input'] = input_list[idx]

                alpaca_dict = {
                    'system': alpaca_dict.get('system', ''),
                    'instruction': alpaca_dict.get('instruction', ''),
                    'input': alpaca_dict.get('input', ''),
                    'output': alpaca_dict.get('output', '')
                }
            else:
                alpaca_dict = {
                    'system': alpaca_dict.get('system', ''),
                    'instruction': alpaca_dict.get('instruction', ''),
                    'output': alpaca_dict.get('output', '')
                }

            alpaca_data.append(alpaca_dict)

        else:
            # Multi Turn
            text = ""
            for role_content in message:
                if role_content['role'] == "system":
                    system_message = role_content['content']
                    text += system_message
                elif role_content['role'] == "user":
                    text += "\n\n### Instruction: " + role_content['content']
                    if input_list:
                        text+= "\n\n### Input: " + input_list[idx]
                elif role_content['role'] == "assistant":
                    text += "\n\n### Response: " + role_content['content']
                else:
                    raise ValueError("Only 'system', 'user', and 'assistant' roles are allowed.")
                
            alpaca_data.append({"text": text})

    if save_file:
        save_jsonl(alpaca_data, filename=save_file)
        
    return alpaca_data