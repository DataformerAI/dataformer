# Embedding Component Documentation

## Overview
The `AsyncLLM` component is used to generate embeddings for a given input using a specified model. This documentation provides an example of how to use the component to obtain embeddings from an external API.

## Example Usage

### Code Example
```python
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize AsyncLLM with the API URL and model
llm = AsyncLLM(url="https://api.deepinfra.com/v1/openai/embeddings", model="thenlper/gte-large")

# Define the instruction for which to generate an embedding
instruction = "hey" 
data_dict = {
    "input": instruction,
    # "encoding_format": "float"  # Optional: specify encoding format if needed
}

# Create a list of requests
request_list = [data_dict]

# Generate embeddings
response_list = llm.generate(request_list)

# Print the response
print(response_list)
```

### Example Input
The input data consists of a dictionary with the following key:
- `input`: A string representing the instruction for which the embedding is to be generated.

For example:
```python
data_dict = {
    "input": "hey"
}
```

### Example Output
The output will be a list of responses containing the generated embeddings. Each response typically includes the embedding vector corresponding to the input instruction.

For example, the output might look like this:
```python
[
    {
        "embedding": [0.123, -0.456, 0.789, ...],  # Example embedding vector
        "input": "hey"
    }
]
```

### Note
- Ensure that the environment variables required for the API are correctly set in the `.env` file.
- The actual embedding values will depend on the model used and the input provided.
