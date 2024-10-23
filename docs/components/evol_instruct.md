## EvolInstruct Component

The `EvolInstruct` class is the core component responsible for evolving instructions. It takes the following input parameters:

- `llm: AsyncLLM`: The language model used for generating evolved instructions.
- `num_evolutions: int`: The number of times to evolve each instruction (default is 1).
- `store_evolutions: bool`: Whether to store the evolved instructions (default is False).
- `generate_answers: bool`: Whether to generate answers for the evolved instructions (default is False).
- `include_original_instruction: bool`: Whether to include the original instruction in the output (default is False).
- `mutation_templates: Dict[str, str]`: A dictionary of templates used for mutating instructions (default is `MUTATION_TEMPLATES`).

### Recommended Models

For optimal performance, consider the following models:

**OpenAI Models**:
  - `gpt-4-turbo`
  - `gpt-4`
  - `gpt-4o-mini`
  - `gpt-4o`
  - `openai-o1-Preview`
  
**MonsterAPI Model**:
  - `google/gemma-2-9b-it`
  
**GROQ Models**:
  - `gemma2-9b-it`
  - `mixtral-8x7b-32768`
  
**DeepInfra Model**:
  - `Qwen/Qwen2.5-72B-Instruct`

### Methods

#### `evolve_instruction`

```python
def evolve_instruction(self, instruction: str) -> List[str]:
    """Evolves an instruction based on the mutation templates."""
```

- **Parameters**: 
  - `instruction` (str): The instruction to be evolved.
- **Returns**: 
  - A list of evolved instructions.

#### `generate`

```python
def generate(self, instructions, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Generates evolved instructions for a list of instructions."""
```

- **Parameters**: 
  - `instructions` (List[str]): A list of instructions to evolve.
  - `use_cache` (bool): Whether to use cached results (default is True).
- **Returns**: 
  - A list of dictionaries containing original and evolved instructions, and optionally answers.

### Example Code for EvolInstruct

Here is an example of how to initialize and use the `EvolInstruct` class:

```python
from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM

# Initialize the language model
llm = AsyncLLM(model="gpt-4-turbo", api_provider="openai")

# Create an instance of EvolInstruct
evol_instruct = EvolInstruct(
    llm=llm,
    num_evolutions=3,
    store_evolutions=True,
    generate_answers=True,
)

# Example usage
instructions = ["What is the capital of France?", "Explain quantum mechanics."]
results = evol_instruct.generate(instructions)

# Display results
for item in results:
    print(item)
```

### Example Input and Output

**Input:**
```python
instructions = ["What is the capital of France?", "Explain quantum mechanics."]
results = evol_instruct.generate(instructions)
```

**Output:**
```json
{
    "original_instruction": "What is the capital of France?",
    "evolved_instructions": [
        "What city serves as the capital of France?",
        "Can you tell me the capital city of France?",
        "What is the name of France's capital?"
    ],
    "answers": [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The capital city of France is Paris."
    ]
}
{
    "original_instruction": "Explain quantum mechanics.",
    "evolved_instructions": [
        "Can you provide an explanation of quantum mechanics?",
        "What is quantum mechanics?",
        "Describe the principles of quantum mechanics."
    ],
    "answers": [
        "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales.",
        "It explains the behavior of matter and energy on atomic and subatomic levels.",
        "Quantum mechanics is essential for understanding the behavior of particles."
    ]
}
```
