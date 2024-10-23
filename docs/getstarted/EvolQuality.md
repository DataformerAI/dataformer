This document outlines the implementation of the **Evolution Quality** component using various language models. The component is designed to generate answers based on instructions and then evolve those responses for quality improvement.

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

---

## Dataset Loading and Response Generation

To load the dataset, generate responses, and evolve those responses for quality improvement, use the following code:

### Step 1: Import Required Libraries

```python
from dataformer.components.evol_quality import EvolQuality
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv
```

### Step 2: Load Environment Variables

```python
# Load environment variables from .env file
load_dotenv()
```

### Step 3: Load the Dataset

```python
dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(2))
instructions = [example["question"] for example in datasetsub]
```

### Step 4: Define Color Constants

```python
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
```

### Step 5: Initialize the Language Model

```python
llm = AsyncLLM(
    model="mixtral-8x7b-32768", api_provider="groq"
)  # Ensure "GROQ_API_KEY" is set in .env file.
```

### Step 6: Generate Answers for the Questions

```python
# Generating answers for the questions
request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
]
answers = llm.generate(request_list, use_cache=True)
answers = [answer[1]["choices"][0]["message"]["content"] for answer in answers]
```

### Step 7: Format Inputs for EvolQuality

```python
# Formatting inputs for EvolQuality
inputs = [
    {"instruction": instruction, "response": response}
    for instruction, response in zip(instructions, answers)
]
```

### Step 8: Initialize EvolQuality and Generate Results

```python
evol_quality = EvolQuality(
    llm=llm,
    num_evolutions=1,  # Number of times to evolve each response
    store_evolutions=True,  # Store all evolutions
    include_original_response=False,  # Exclude original response in evolved_responses
)
results = evol_quality.generate(inputs, use_cache=False)
```

### Step 9: Display Results

```python
print("\n\n")
for result in results:
    print(f"{COLOR['BLUE']}Instruction: {result['instruction']}{COLOR['ENDC']}")
    print(f"{COLOR['GREEN']}Response: {result['response']}{COLOR['ENDC']}")
    for evolved_response in result["evolved_responses"]:
        print(f"{COLOR['PURPLE']}Evolved Response: {evolved_response}{COLOR['ENDC']}")
    print("\n")
```

## Code Summary

Here is the complete code for the Evolution Quality component:

```python
from dataformer.components.evol_quality import EvolQuality
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(2))
instructions = [example["question"] for example in datasetsub]

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

llm = AsyncLLM(
    model="mixtral-8x7b-32768", api_provider="groq"
)  # Ensure "GROQ_API_KEY" is set in .env file.

# Generating answers for the questions
request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
]
answers = llm.generate(request_list, use_cache=True)
answers = [answer[1]["choices"][0]["message"]["content"] for answer in answers]

# Formatting inputs for EvolQuality
inputs = [
    {"instruction": instruction, "response": response}
    for instruction, response in zip(instructions, answers)
]

evol_quality = EvolQuality(
    llm=llm,
    num_evolutions=1,  # Number of times to evolve each response
    store_evolutions=True,  # Store all evolutions
    include_original_response=False,  # Exclude original response in evolved_responses
)
results = evol_quality.generate(inputs, use_cache=False)

print("\n\n")
for result in results:
    print(f"{COLOR['BLUE']}Instruction: {result['instruction']}{COLOR['ENDC']}")
    print(f"{COLOR['GREEN']}Response: {result['response']}{COLOR['ENDC']}")
    for evolved_response in result["evolved_responses"]:
        print(f"{COLOR['PURPLE']}Evolved Response: {evolved_response}{COLOR['ENDC']}")
    print("\n")
```

## Code Explanation

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
The `EvolQuality` component evolves instructions a specified number of times, storing results and generating answers for comprehensive analysis.

## Usage
To use the Evolution Quality component, ensure that you have the required dependencies installed and the environment variables set up correctly. Then, follow the code example provided to load your dataset, generate responses, and evolve them for quality improvement.

## Results Display
Results are printed with color-coded formatting for easy readability, distinguishing between original and evolved instructions and their answers.
