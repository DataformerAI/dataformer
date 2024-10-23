# Pvg Class Documentation

## Overview
The `pvg` class implements a Problem Verification Game (PVG) approach for generating and refining solutions to problems using a language model (LLM). It allows for iterative solution generation, verification, and refinement based on user queries.

## Initialization
### `__init__(self, llm, num_rounds: int = 3, num_solutions: int = 2, verify_model="meta-llama/Meta-Llama-3.1-8B-Instruct")`
- **Parameters**:
  - `llm`: An instance of a language model used for generating responses.
  - `num_rounds`: The number of rounds for generating and verifying solutions (default is 3).
  - `num_solutions`: The number of solutions to generate in each round (default is 2).
  - `verify_model`: The model used for verification (default is "meta-llama/Meta-Llama-3.1-8B-Instruct").
- **Description**: Initializes the `pvg` class with the provided language model and parameters.

## Methods

### `generate(self, request_list, return_model_answer=True)`
- **Parameters**:
  - `request_list`: A list of requests to be processed.
  - `return_model_answer`: A boolean flag indicating whether to return the model's answer.
- **Returns**: A list of dictionaries containing the model's response and the PVG response.
- **Description**: Generates responses based on the provided requests. If `return_model_answer` is true, it retrieves the model's response and combines it with the PVG reflection.

### `generate_solutions(self, request_list, request_list_modified, num_solutions: int, is_sneaky: bool = False, temperature: float = 0.7)`
- **Parameters**:
  - `request_list`: The original list of requests.
  - `request_list_modified`: The modified list of requests for generating solutions.
  - `num_solutions`: The number of solutions to generate.
  - `is_sneaky`: A boolean flag indicating whether to generate "sneaky" solutions (default is False).
  - `temperature`: A float value controlling the randomness of the output (default is 0.7).
- **Returns**: A list of generated solutions.
- **Description**: Generates solutions based on the provided requests, either in "helpful" or "sneaky" mode.

### `verify_solutions(self, system_prompt, initial_query, solutions)`
- **Parameters**:
  - `system_prompt`: The system prompt for the verification process.
  - `initial_query`: The original query for which solutions are being verified.
  - `solutions`: A list of solutions to be verified.
- **Returns**: A list of scores for each solution.
- **Description**: Verifies the correctness and clarity of the provided solutions, returning a score for each.

### `gather_requests(self, request_list)`
- **Parameters**:
  - `request_list`: A list of requests containing messages.
- **Returns**: A modified list of requests with system prompts and initial queries.
- **Description**: Processes the input requests to extract system prompts and user/assistant messages, formatting them for further processing.

### `pvg(self, request_list)`
- **Parameters**:
  - `request_list`: A list of requests to be processed.
- **Returns**: A list of the best solutions found after the verification rounds.
- **Description**: Implements the PVG process, generating solutions, verifying them, and refining queries over multiple rounds.

## Usage Example

### Example Input
```python
from dataformer.components.pvg import pvg
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = AsyncLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_provider="deepinfra"
)

# Example request for the pvg class
request_list = [
    {"messages": [{"role": "user", "content": "How can I optimize a sorting algorithm?"}]} 
]

# Create an instance of the pvg class
pvg_instance = pvg(llm=llm)
results = pvg_instance.generate(request_list)

# Print the results
print("\n\n")
print(f"Prompt: {request_list[0]['messages'][0]['content']}")
print("\n")
for item in results:
    print(f"PVG Answer: {item['pvg_response']}")
    print(f"Model Answer: {item['model_response']}")
    print("\n")
```

```
Prompt: How can I optimize a sorting algorithm?

PVG Answer: 
To optimize a sorting algorithm, consider the following strategies:
1. **Choose the Right Algorithm**: Depending on the data size and characteristics, choose an appropriate sorting algorithm (e.g., QuickSort for average cases, MergeSort for stability).
2. **Use Hybrid Approaches**: Combine different algorithms for different data sizes (e.g., use Insertion Sort for small arrays).
3. **Reduce Comparisons**: Implement techniques like counting sort or radix sort for specific cases where the range of input values is limited.
4. **Parallel Processing**: Utilize multi-threading or distributed computing to sort large datasets more efficiently.
5. **In-Place Sorting**: Use algorithms that require minimal additional space to reduce memory overhead.

Model Answer: The best way to optimize a sorting algorithm depends on the specific use case and data characteristics. Consider the above strategies to improve performance.
```