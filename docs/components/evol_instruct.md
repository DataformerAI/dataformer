# Evolution Instruction Component

This document outlines the implementation of the Evolution Instruction component using various language models. The component is designed to evolve instructions based on a dataset and generate corresponding answers.

## Dependencies

Ensure you have the following dependencies installed:

- `dataformer`
- `datasets`
- `python-dotenv`
- `logging`

## Setup

### Load Environment Variables

Load environment variables from a `.env` file to manage sensitive information such as API keys.

### Recommended Models

Here are some recommended models for optimal performance:

- **OpenAI Models**:
  - `gpt-4-turbo`
  - `gpt-4`
  - `gpt-4o-mini`
  - `gpt-4o`
  - `openai-o1-Preview`

- **MonsterAPI Model**:
  - `google/gemma-2-9b-it`

- **GROQ Model**:
  - `gemma2-9b-it`
  - `mixtral-8x7b-32768`

- **DeepInfra Model**:
  - `Qwen/Qwen2.5-72B-Instruct`

## Dataset Loading

To load the dataset and select a subset for instruction evolution, you can use the following code:

```python
from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define terminal output colors
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

# Load the dataset and select a subset of the data
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(2))
instructions = [example["question"] for example in datasetsub]

# Initialize the model with the chosen API provider and model
llm = AsyncLLM(
    model="mixtral-8x7b-32768", api_provider="groq"
)  # Ensure "GROQ_API_KEY" is set in the .env file.

# Configure the Evolution Instruction component
evol_instruct = EvolInstruct(
    llm=llm,
    num_evolutions=2,  # Number of times to evolve each instruction
    store_evolutions=True,  # Store all instruction evolutions
    generate_answers=True,  # Generate answers for the evolved instructions
    # include_original_instruction=True  # Optionally include the original instruction in results
)

# Generate evolved instructions and answers
results = evol_instruct.generate(
    instructions, use_cache=False  # Disable cache for fresh results
)

# Print the results in a formatted way
print("\n\n")
for item in results:
    print(f"{COLOR['BLUE']}Original Instruction: {item['original_instruction']}{COLOR['ENDC']}")
    for evolved_instruction, answer in zip(item['evolved_instructions'], item['answers']):
        print(f"{COLOR['GREEN']}Evolved Instruction: {evolved_instruction}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Answer: {answer}{COLOR['ENDC']}")
    print("\n")
```

## Explanation of the Code

### 1. Loading Environment Variables
The `.env` file securely loads sensitive information, such as API keys, keeping credentials safe and separate from the codebase.

### 2. Model Setup
Available models include:
- **OpenAI Models**: High-performance models for coherent responses.
- **MonsterAPI Models**: Specialized capabilities for specific tasks.
- **GROQ Models**: Optimized for instruction evolution.

Select the model that best fits your needs.

### 3. Dataset Loading
The dataset is sourced from `dataformer/self-knowledge` using the `datasets` library, with a subset selected for efficient testing and experimentation.

### 4. Instruction Evolution
The `EvolInstruct` component evolves instructions a specified number of times, storing results and generating answers for comprehensive analysis.

### 5. Results Display
Results are printed with color-coded formatting for easy readability, distinguishing between original and evolved instructions and their answers.
