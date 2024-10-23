## EvolQuality Component

The `EvolQuality` class is responsible for evolving responses based on given instructions. It takes the following input parameters:

- `llm: AsyncLLM`: The language model used for generating evolved responses.
- `num_evolutions: int`: The number of times to evolve each response (default is 1).
- `store_evolutions: bool`: Whether to store the evolved responses (default is False).
- `include_original_response: bool`: Whether to include the original response in the output (default is False).
- `mutation_templates: Dict[str, str]`: A dictionary of templates used for mutating responses (default is `MUTATION_TEMPLATES`).

### Recommended Models

For optimal performance, consider the following models:

**OpenAI Models**:
  - `gpt-3.5-turbo`
  - `gpt-4-turbo`
  - `gpt-4`
  - `gpt-4o-mini`
  - `gpt-4o`
  - `o1-mini`
  - `o1-preview`

**MonsterAPI Model**:
  - `google/gemma-2-9b-it`
  - `microsoft/Phi-3-mini-4k-instruct`

**GROQ Model**:
  - `gemma2-9b-it`
  - `mixtral-8x7b-32768`

**DeepInfra Model**:
  - `meta-llama/Meta-Llama-3.1-405B-Instruct`
  - `microsoft/WizardLM-2-8x22B`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `Qwen/Qwen2.5-72B-Instruct`


### Methods

#### `evolve_responses`

```python
def evolve_responses(self, instruction: str, response: str) -> List[str]:
    """Evolves a response based on the mutation templates."""
```

- **Parameters**: 
  - `instruction` (str): The instruction associated with the response.
  - `response` (str): The original response to be evolved.
- **Returns**: 
  - A list of evolved responses.

#### `generate`

```python
def generate(self, inputs, use_cache: bool = True):
    """Generates evolved responses for a list of inputs containing instructions and responses."""
```

- **Parameters**: 
  - `inputs` (List[Dict[str, str]]): A list of dictionaries containing instructions and responses.
  - `use_cache` (bool): Whether to use cached results (default is True).
- **Returns**: 
  - A list of dictionaries containing the original instruction, original response, and evolved responses.

### Example Code for EvolQuality

Here is an example of how to initialize and use the `EvolQuality` class:

```python
from dataformer.components.evol_quality import EvolQuality
from dataformer.llms import AsyncLLM

# Initialize the language model
llm = AsyncLLM(model="gpt-4-turbo", api_provider="openai")

# Create an instance of EvolQuality
evol_quality = EvolQuality(
    llm=llm,
    num_evolutions=3,
    store_evolutions=True,
    include_original_response=True,
)

# Example usage
inputs = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Explain quantum mechanics.", "response": "Quantum mechanics is a fundamental theory in physics."}
]
results = evol_quality.generate(inputs)

# Display results
for item in results:
    print(item)
```

### Example Input and Output

**Input:**
```python
inputs = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Explain quantum mechanics.", "response": "Quantum mechanics is a fundamental theory in physics."}
]
results = evol_quality.generate(inputs)
```

**Output:**
```json
{
    "instruction": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "evolved_responses": [
        "Paris is the capital city of France.",
        "The city of Paris serves as the capital of France.",
        "France's capital is Paris."
    ]
}
{
    "instruction": "Explain quantum mechanics.",
    "response": "Quantum mechanics is a fundamental theory in physics.",
    "evolved_responses": [
        "Quantum mechanics describes the behavior of matter and energy at the smallest scales.",
        "It is a theory that explains the nature of particles and their interactions.",
        "Quantum mechanics is essential for understanding atomic and subatomic processes."
    ]
}
```