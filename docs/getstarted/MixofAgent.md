# Mixture of Agents Documentation

## Overview
The Mixture of Agents technique is designed to achieve superior performance and results by employing a layered architecture. In this approach, multiple language models (LLMs) are utilized to generate responses to user queries, which are then synthesized by an aggregator model. This documentation outlines the implementation of a two-layered approach where the first layer consists of various LLMs generating answers, and the second layer aggregates these responses into a single, coherent reply.

## Example Usage

### Code Example
```python
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the API keys
deepinfra_api_key = ""  # Add your DeepInfra API key here
openai_api_key = ""     # Add your OpenAI API key here

# Define the reference models, their providers, and keys for layer 1
reference_models_providers = {
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ["deepinfra", deepinfra_api_key],
    "gpt-4o": ["openai", openai_api_key]
}

# Colors for printing the output
COLOR = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "ENDC": "\033[0m",
}

# Define the aggregator model and system prompt
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models
to the latest user query. Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it
may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined,
accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and
adheres to the highest standards of accuracy and reliability.

Responses from models: """

# Specify the API provider for the aggregator model
api_provider = "deepinfra"

# Define the aggregator LLM
aggregator_llm = AsyncLLM(api_provider=api_provider, model=aggregator_model)

# Define user prompts
request_list = [
    {
        "messages": [
            {
                "role": "user",
                "content": "Give only names of 3 places to visit in India."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Give only names of any 3 fiction books."
            }
        ]
    }
]

# Creating AsyncLLM object to provide different model responses to the user query
llm = AsyncLLM()

# Creating new requests list with user queries and required models
final_request_list = []
for models in reference_models_providers:
    for request in request_list:
        new = request.copy()
        new["model"] = models  # Adding the respective model
        new["api_provider"] = reference_models_providers[models][0]
        new["api_key"] = reference_models_providers[models][1]
        final_request_list.append(new)

# Collect responses from the reference LLMs
reference_models_response_list = llm.generate(final_request_list)

# Store the processed responses for passing to the aggregator LLM
reference_models_results = []

print(f"{COLOR['RED']}Models Individual Responses{COLOR['ENDC']}")

# Create reference_models_results for storing all models' responses
for i in range(len(reference_models_providers)):
    reference_models_results.append([])

# Iterating over the responses
model_incr = 0
for i in range(0, len(reference_models_response_list), len(reference_models_providers)):
    answer_incr = 0
    response_list = reference_models_response_list[i:i + len(reference_models_providers)]
    
    for request, response in zip(request_list, response_list):
        prompt = request["messages"][0]["content"]
        answer = response[1]["choices"][0]["message"]["content"]
        print(f"{COLOR['BLUE']}Reference Model: {model_incr}\n Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")
        
        # Store model's responses to a query
        reference_models_results[answer_incr].append(str(model_incr) + "... " + answer)
        answer_incr += 1
    model_incr += 1

# Pass the responses of models to the aggregator LLM
request_list_aggregator = []
for i in range(len(request_list)):
    request_list_aggregator.append({
        "messages": [
            {
                "role": "system",
                "content": aggregator_system_prompt + "\n" + "\n".join(reference_models_results[i])
            },
            {
                "role": "user",
                "content": request_list[i]["messages"][0]["content"]
            }
        ]
    })

# Generate the response from the aggregator LLM
response_list_aggregator = aggregator_llm.generate(request_list_aggregator)

# Print the response from the aggregator LLM
print(f"{COLOR['RED']}Aggregator Model's Response{COLOR['ENDC']}")
for request, response in zip(request_list, response_list_aggregator):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"{COLOR['BLUE']}Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")

```
### Example Input
The input consists of user queries defined in the `request_list`. For example:
1. "Give only names of 3 places to visit in India."
2. "Give only names of any 3 fiction books."

### Example Output
The output will include individual responses from each reference model and the final synthesized response from the aggregator model. For example:
```
Models Individual Responses
Reference Model: 0
 Prompt: Give only names of 3 places to visit in India.
Answer:
 1. Taj Mahal
 2. Jaipur
 3. Goa

Reference Model: 1
 Prompt: Give only names of 3 places to visit in India.
Answer:
 1. Delhi
 2. Kerala
 3. Mumbai

Aggregator Model's Response
Prompt: Give only names of 3 places to visit in India.
Answer:
 1. Taj Mahal
 2. Jaipur
 3. Goa
```

### Note
- Ensure that the environment variables required for the API are correctly set in the `.env` file.
- The actual responses will depend on the models used and the input provided.
