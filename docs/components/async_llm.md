# Asynchronous LLM Usage

This document provides an overview of how to use the asynchronous LLM (Large Language Model) from the `dataformer` library.

## Setup

Before using the asynchronous LLM, ensure you have the necessary environment variables set up. You can do this by creating a `.env` file in your project directory.

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
```

## Importing Required Libraries

You will need to import the following libraries:

```python
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list
from datasets import load_dataset
```

## Loading the Dataset

Load the dataset you want to use with the LLM. In this example, we are using a self-knowledge dataset.

```python
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(5))
instructions = [example["question"] for example in datasetsub]
```

## Preparing Requests

Define the sampling parameters and prepare the request list for the LLM.

```python
sampling_params = {"temperature": 0.7}
request_list = get_request_list(instructions, sampling_params)
```

## Using the Asynchronous LLM

Instantiate the `AsyncLLM` with your chosen API provider and generate responses.

```python
llm = AsyncLLM(api_provider="groq")
response_list = llm.generate(request_list)
```

## Displaying Results

Finally, iterate through the requests and responses to display the prompts and answers.

```python
for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")
```

## API Providers

Supported API providers include:
- OpenAI
- Groq
- Together
- DeepInfra
- OpenRouter

## Full Code Example

Here is the complete code for using the asynchronous LLM:

```python
from dotenv import load_dotenv
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

# Load the dataset
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(5))
instructions = [example["question"] for example in datasetsub]

# Prepare requests
sampling_params = {"temperature": 0.7}
request_list = get_request_list(instructions, sampling_params)

# Instantiate the AsyncLLM
llm = AsyncLLM(api_provider="groq")
response_list = llm.generate(request_list)

# Display results
for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")
```
