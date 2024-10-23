# Text Generation with Together API

## Overview
This documentation provides a guide on how to use the `AsyncLLM` class from the `dataformer.llms` module to generate text using the Together API. The example demonstrates how to load environment variables, create a request list, and generate responses based on user prompts.

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
   Create a file named `.env` in your project directory. This file will store your environment variables, such as API keys.

2. **Load Environment Variables**: 
   Use the `load_dotenv()` function to load the variables from the `.env` file. This is essential for managing sensitive information.

3. **Define Your Requests**: 
   Create a list of prompts that you want the model to respond to. Each prompt should be structured as a dictionary.

## Example Code
Below is a sample code snippet to demonstrate how to generate text using the Together API:

```python
from dotenv import load_dotenv
from dataformer.llms import AsyncLLM

# Load environment variables from .env file
load_dotenv()

# Define the request list with prompts
request_list = [
    {"prompt": "Complete the paragraph.\n She lived in Nashville."},
    {"prompt": "Write a story on 'Honesty is the best Policy'."}
]

# Initialize the AsyncLLM object with the Together API provider
llm = AsyncLLM(api_provider="together", gen_type="text")

# Generate responses based on the request list
response = llm.generate(request_list)

# Print the generated responses
print(response)
```

## Example Output
When you run the above code, you can expect output similar to the following (the actual output will depend on the model's responses):

```
[
    {
        "prompt": "Complete the paragraph.\n She lived in Nashville.",
        "completion": "She lived in Nashville, a city known for its vibrant music scene and rich cultural heritage. Every evening, she would stroll down Broadway, soaking in the sounds of live country music and the warmth of Southern hospitality."
    },
    {
        "prompt": "Write a story on 'Honesty is the best Policy'.",
        "completion": "Once upon a time in a small village, there lived a young boy named Sam. Sam was known for his honesty, even when it was difficult. One day, he found a lost wallet filled with money. Instead of keeping it, he returned it to its owner, who was so grateful that he rewarded Sam with a special gift. This act of honesty taught the villagers that being truthful always brings good fortune."
    }
]
```

## Conclusion
By following the steps outlined in this documentation, you can successfully set up and use the Together API to generate text based on user-defined prompts. Make sure to replace any placeholder values in your `.env` file with your actual API keys and configurations.
