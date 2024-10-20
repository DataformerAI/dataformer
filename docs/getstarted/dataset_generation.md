# Quick Start

Welcome to the Quick Start guide for Dataformer's AsyncLLM! This guide will help you quickly set up and generate responses asynchronously using various API providers.

## Supported API Providers
Dataformer's AsyncLLM supports the following API providers:
- **OpenAI**
- **Groq**
- **Together**
- **DeepInfra**
- **OpenRouter**

Choose the provider that best suits your needs!

## Example: Generating Responses Asynchronously

Here's a quick example of how to use Dataformer's AsyncLLM for efficient asynchronous generation of responses:

```python
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list, get_messages
from datasets import load_dataset

# Load a sample dataset
dataset = load_dataset("dataformer/self-knowledge", split="train").select(range(3))
instructions = [example["question"] for example in dataset]

# Prepare the request list with sampling parameters
sampling_params = {"temperature": 0.7}
request_list = get_request_list(instructions, sampling_params)

# Initialize AsyncLLM with your preferred API provider
# {{ edit_1 }}: Added API key handling
llm = AsyncLLM(api_provider="groq", model="llama-3.1-8b-instant")

# Generate responses asynchronously
response_list = get_messages(llm.generate(request_list))

# Output the generated responses
for response in response_list:
    print(response)
```

### Explanation of the Code:
1. **Import Necessary Libraries**: The code begins by importing the required modules from Dataformer and the datasets library.
2. **Load a Sample Dataset**: A sample dataset is loaded, and a few example questions are extracted for processing.
3. **Prepare the Request List**: The instructions are prepared into a request list with specified sampling parameters (like temperature).
4. **Initialize AsyncLLM**: An instance of AsyncLLM is created, specifying the API provider and model. You can also create a `.env` file to store your API keys:
   ```
   OPENAI_API_KEY=
   GROQ_API_KEY=
   TOGETHER_API_KEY= 
   ANYSCALE_API_KEY=
   DEEPINFRA_API_KEY=
   OPENROUTER_API_KEY=
   MONSTER_API_KEY=
   ANTHROPIC_API_KEY=
   ```
5. **Generate Responses**: The `generate` method is called to produce responses asynchronously based on the request list.
6. **Output the Responses**: Finally, the generated responses are printed to the console.

This example demonstrates how easy it is to get started with Dataformer's AsyncLLM for generating responses based on your dataset. Feel free to modify the parameters and explore different API providers to suit your needs!
