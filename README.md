<div align="center">
  <img src="https://github.com/DataformerAI/dataformer/assets/39311993/b2515523-19a9-4a54-8f12-1f8de24b7a9f"/>
</div>

<h3 align="center">Solving data for LLMs - Create quality synthetic datasets!</h3>

<p align="center">
  <a href="https://x.com/dataformer_ai">
    <img src="https://img.shields.io/badge/twitter-black?logo=x"/>
  </a>
  <a href="https://www.linkedin.com/company/dataformer">
    <img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
  </a>
  <a href="https://dataformer.ai/discord">
    <img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
  </a>
  <a href="https://dataformer.ai/call">
    <img src="https://img.shields.io/badge/book_a_call-00897B?&logo=googlemeet&logoColor=white"/>
  </a>
</p>

## Why Dataformer?

Dataformer empowers engineers with a robust framework for creating high-quality synthetic datasets for AI, offering speed, reliability, and scalability. Our mission is to supercharge your AI development process by enabling rapid generation of diverse, premium datasets grounded in proven research methodologies. 
In the world of AI, compute costs are high, and output quality is paramount. Dataformer allows you to **prioritize data excellence**, addressing both these challenges head-on. By crafting top-tier synthetic data, you can invest your precious time in achieving and sustaining **superior standards** for your AI models.

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

## Join Community

[Join Dataformer on Discord](https://dataformer.ai/discord)
