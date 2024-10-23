# Ollama Integration Documentation

## Overview
Ollama provides an OpenAI-compatible endpoint that allows you to interact with language models seamlessly. This documentation outlines the steps to set up your environment and use the `AsyncLLM` class from the `dataformer.llms` module to communicate with the Ollama API.

## Prerequisites
Before you begin, ensure you have the following:
- Python installed on your machine.
- The `dataformer` library installed. You can install it using pip:
  ```bash
  pip install dataformer
  ```
- The `python-dotenv` library to manage environment variables:
  ```bash
  pip install python-dotenv
  ```

## Setup Instructions

1. **Create a `.env` File**: 
   Create a file named `.env` in your project directory. This file will store your environment variables. 

2. **Load Environment Variables**: 
   Use the `load_dotenv()` function to load the variables from the `.env` file. This is essential for managing sensitive information like API keys.

3. **Get Your Ollama Endpoint URL**:
   - Visit the [Ollama Template Page](https://jarvislabs.ai/templates/ollama) to create an instance.
   - After creating an instance, click on the API section to retrieve your Endpoint URL.
   - Follow the deployment guide at [Ollama Deployment Guide](https://jarvislabs.ai/blogs/ollama_deploy) for instructions on how to deploy your model (e.g., `ollama pull llama3`).

4. **Set Up Your Code**:
   Below is a sample code snippet to get you started with the Ollama API:

   ```python
   from dataformer.llms import AsyncLLM
   from dotenv import load_dotenv

   # Load environment variables from .env file
   load_dotenv()

   # Ollama - OpenAI compatible endpoint Example Url
   URL = "https://a8da29c1850e1.notebooksa.jarvislabs.net/v1/chat/completions"

   # Define sampling parameters
   sampling_params = {"temperature": 0.6, "top_p": 1}

   # Initialize the AsyncLLM object
   llm = AsyncLLM(model="llama3", url=URL, sampling_params=sampling_params, api_provider="ollama", max_requests_per_minute=5)

   # Define user requests
   request_list = [
       {"messages": [{"role": "user", "content": "Hi there!"}], "stream": False},
       {"messages": [{"role": "user", "content": "Who are you?"}], "stream": False}
   ]

   # Generate responses
   response_list = llm.generate(request_list)

   # Print the responses
   for request, response in zip(request_list, response_list):
       prompt = request["messages"][0]["content"]
       answer = response[1]["choices"][0]["message"]["content"]
       print(f"Prompt: {prompt}\nAnswer: {answer}")
   ```

## Example Output
When you run the above code, you can expect output similar to the following:
```
Prompt: Hi there!
Answer: Hello! How can I assist you today?

Prompt: Who are you?
Answer: I am an AI language model here to help you with your queries.
```

