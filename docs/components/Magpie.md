# MAGPIE Class Documentation

## Overview
The `MAGPIE` class is designed to facilitate the generation of question-answer pairs using a language model (LLM). It allows for customizable templates and supports multiple languages, making it versatile for various applications.

## Initialization
### `__init__(self, llm, template=None, lang="en")`
- **Parameters**:
  - `llm`: An instance of a language model used for generating responses.
  - `template`: An optional string template for the queries. If not provided, a default template based on the model will be used.
  - `lang`: The language for the queries (default is "en" for English).
- **Description**: Initializes the `MAGPIE` class with the specified language model, template, and language.

## Methods

### `create_requests(self, prompt, role="user")`
- **Parameters**:
  - `prompt`: The prompt to be sent to the language model.
  - `role`: The role of the message sender (default is "user").
- **Returns**: A dictionary containing the model, stream status, and messages.
- **Description**: Constructs a request dictionary for the language model based on the provided prompt and role.

### `extract(self, text)`
- **Parameters**:
  - `text`: A string containing the text to be processed.
- **Returns**: The first non-empty line of the text, stripped of whitespace.
- **Description**: Extracts the first meaningful line from the provided text.

### `validate(self, entry)`
- **Parameters**:
  - `entry`: A dictionary containing a question and answer.
- **Returns**: The entry if valid; otherwise, returns `False`.
- **Description**: Validates the entry to ensure it contains a question and a non-empty answer.

### `display(self, num_samples)`
- **Parameters**:
  - `num_samples`: The number of samples to be generated.
- **Description**: Displays the parameters for the dataset creation, including model, total samples, language, and query template.

### `generate(self, num_samples, use_cache=False)`
- **Parameters**:
  - `num_samples`: The number of question-answer pairs to generate.
  - `use_cache`: A boolean flag indicating whether to use cached responses (default is `False`).
- **Returns**: A list of dictionaries containing validated question-answer pairs.
- **Description**: Generates the specified number of question-answer pairs by creating requests, processing responses, and validating the results.

## Usage Example

### Example Input
```python
from dataformer.llms import AsyncLLM
from dataformer.components.magpie.prompts import languages, templates
from dataformer.components.magpie import MAGPIE
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = AsyncLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_provider="deepinfra"
)

# Example template for MAGPIE
templates = {
    "llama3": "Generate a question and answer based on the following context: What is the captial of France? ",
}

# Create an instance of the MAGPIE class
magpie_instance = MAGPIE(llm=llm, template=templates["llama3"])

# Generate question-answer pairs
num_samples = 5
dataset = magpie_instance.generate(num_samples)

# Print the generated dataset
for entry in dataset:
    print(f"Question: {entry['question']}")
    print(f"Answer: {entry['answer']}\n")
```

### Example Output
````
Creating dataset with the following parameters:
MODEL: meta-llama/Meta-Llama-3.1-8B-Instruct
Total Samples: 5
Language: English
Query Template: [Your template here]

Question: What is the capital of France?
Answer: The capital of France is Paris.

Question: How does photosynthesis work?
Answer: Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll.

Question: What is the Pythagorean theorem?
Answer: The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides.
````

## Conclusion
The `MAGPIE` class provides a structured approach to generating question-answer pairs using a language model. It supports customizable templates and multiple languages, making it a valuable tool for various applications in natural language processing and AI-driven content generation.
