<div align="center">
  <img src="https://github.com/DataformerAI/dataformer/assets/39311993/b2515523-19a9-4a54-8f12-1f8de24b7a9f"/>
</div>

<div align="center"  class="margin-top: 20px;" >
  <a href="https://dataformer.ai/discord" style="display: inline-block; background-color: #7289DA; color: white; padding: 4px 8px; margin: 2px; text-decoration: none; font-weight: bold; border-radius: 2px;">
    Join our Discord
  </a>
  <a href="https://dataformer.ai/call" style="display: inline-block; background-color: #00897B; color: white; padding: 4px 8px; margin: 2px; text-decoration: none; font-weight: bold; border-radius: 2px;">
    Book a Call
  </a>
</div>

Dataformer is an open-source library to create high quality synthetic datasets - cheap, fast & reliable data generation at scale backed by research papers.     

You just need a single line of code to use your favourite api provider or local LLM.


<div class="grid cards" markdown>
- 🚀 **Get Started**

    Install with `pip` and get started to generate your first dataset with Dataformer.

    [:octicons-arrow-right-24: Get Started](getstarted/index.md)


- ✅️ **Tutorials**

    Practical guides to help you achieve a specific goals. Learn how to use Dataformer to solve real-world problems.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)


</div>

### One API, Multiple Providers

We integrate with **multiple LLM providers** using one unified API and allow you to make parallel async API calls while respecting rate-limits. We offer the option to cache responses from LLM providers, minimizing redundant API calls and directly reducing operational expenses.

### Research-Backed Iteration at Scale
 
Leverage state-of-the-art research papers to generate synthetic data while ensuring **adaptability, scalability, and resilience**. Shift your focus from infrastructure concerns to refining your data and enhancing your models.

## Installation

PyPi (Stable)
```
pip install dataformer
```

Github Source (Latest):
```
pip install dataformer@git+https://github.com/DataformerAI/dataformer.git 
```

Using Git (Development):
```
git clone https://github.com/DataformerAI/dataformer.git
cd dataformer
pip install -e .
```
## Quick Start

AsyncLLM supports various API providers, including:
- OpenAI
- Groq
- Together
- DeepInfra
- OpenRouter

Choose the provider that best suits your needs!

Here's a quick example of how to use Dataformer's AsyncLLM for efficient asynchronous generation:
```python
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list, get_messages
from datasets import load_dataset

# Load a sample dataset
dataset = load_dataset("dataformer/self-knowledge", split="train").select(range(3))
instructions = [example["question"] for example in dataset]

# Prepare the request list
sampling_params = {"temperature": 0.7}
request_list = get_request_list(instructions, sampling_params)

# Initialize AsyncLLM with your preferred API provider
llm = AsyncLLM(api_provider="groq", model="llama-3.1-8b-instant")

# Generate responses asynchronously
response_list = get_messages(llm.generate(request_list))
```
## Contribute

We welcome contributions! Check our issues or open a new one to get started.