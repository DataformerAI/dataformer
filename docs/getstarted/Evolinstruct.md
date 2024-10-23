## Overview

The **Evolution Instruction** is designed to evolve instructions based on a dataset and generate corresponding answers using various language models. This document outlines the implementation details, dependencies, setup instructions, and usage examples.

## Dependencies

Ensure you have the following dependencies installed:

- `dataformer`
- `datasets`
- `python-dotenv`
- `logging`

## Setup

### Load Environment Variables

To manage sensitive information such as API keys, load environment variables from a `.env` file:

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
```

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

## Dataset Loading

To load the dataset and select a subset for instruction evolution, use the following code:

```python
from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the dataset and select a subset of the data
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(2))
instructions = [example["question"] for example in datasetsub]
```

## Usage Example

Here is a complete example of how to configure and use the Evolution Instruction component:

```python
from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the dataset
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(2))
instructions = [example["question"] for example in datasetsub]

# Initialize the model with the chosen API provider and model
llm = AsyncLLM(model="mixtral-8x7b-32768", api_provider="groq")

# Configure the Evolution Instruction component
evol_instruct = EvolInstruct(
    llm=llm,
    num_evolutions=2,
    store_evolutions=True,
    generate_answers=True,
)

# Generate evolved instructions and answers
results = evol_instruct.generate(instructions, use_cache=False)

# Print the results in a formatted way
for item in results:
    print(f"Original Instruction: {item['original_instruction']}")
    for evolved_instruction, answer in zip(item['evolved_instructions'], item['answers']):
        print(f"Evolved Instruction: {evolved_instruction}")
        print(f"Answer: {answer}")
```

## Code Explanation

### 1. Loading Environment Variables
The `.env` file securely loads sensitive information, such as API keys, keeping credentials safe and separate from the codebase.

### 2. Model Setup
Available models include:
- **OpenAI Models**: High-performance models for coherent responses.
- **MonsterAPI Models**: Specialized capabilities for specific tasks.
- **GROQ Models**: Optimized for instruction evolution.

### 3. Dataset Loading
The dataset is sourced from `dataformer/self-knowledge` using the `datasets` library, with a subset selected for efficient testing and experimentation.

### 4. Instruction Evolution
The `EvolInstruct` component evolves instructions a specified number of times, storing results and generating answers for comprehensive analysis.

### 5. Results Display
Results are printed in a clear format, distinguishing between original and evolved instructions and their answers.

