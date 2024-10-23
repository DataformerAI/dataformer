## Overview

This documentation provides an overview of how to use the `AsyncLLM` class from the `dataformer.llms` module to generate responses from a language model while managing caching effectively. The caching mechanism allows for efficient handling of requests across different projects.

## Prerequisites

Before using the code, ensure you have the following:

- Python installed on your machine.
- The `dataformer` library installed.
- A `.env` file configured with your API credentials.

## Setup

First, load the necessary environment variables from your `.env` file:

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
```

Next, initialize the `AsyncLLM` object:

```python
from dataformer.llms import AsyncLLM

# Initialize object
llm = AsyncLLM(api_provider="together", project_name="generate_Data")
```

## Generating Responses

### Creating Requests

You can create a list of requests to generate responses. Each request should be structured as follows:

```python
request_list = [
    {"messages": [{"role": "user", "content": "Your question here?"}]}
]
```

### Example Requests

**Default Project Name: `dataformer`**

   To generate responses with the default project name, use:

   ```python
   request_list = [
       {"messages": [{"role": "user", "content": "Why should people read books?"}]},
       {"messages": [{"role": "user", "content": "What is the importance of clouds?"}], "api_provider": "together"}
   ]

   llm.generate(request_list)
   ```

**New Project: `Questions`**

   To generate responses for a new project, specify the project name:

   ```python
   request_list = [
       {"messages": [{"role": "user", "content": "Why should people read books?"}]},
       {"messages": [{"role": "user", "content": "Name people who have won medals at olympics 2024."}], "api_provider": "together"}
   ]

   llm.generate(request_list, project_name="Questions")
   ```

**Another New Project: `Maths`**

   You can also create requests for different topics:

   ```python
   request_list = [
       {"messages": [{"role": "user", "content": "What is 2+10/2?"}]},
       {"messages": [{"role": "user", "content": "Solve 5x+2x=0"}], "api_provider": "together"}
   ]

   llm.generate(request_list, project_name="Maths")
   ```

## Managing Cache

### Deleting Cache for Old Projects

To delete the cache for a specific project, you can use:

```python
request_list = [
    {"messages": [{"role": "user", "content": "Why should people read books?"}]},  # Request will be skipped
    {"messages": [{"role": "user", "content": "Name people who have won medals at olympics 2024."}], "api_provider": "together"}
]

# Delete specific project's cache
llm.generate(request_list, project_name="NewProject", clear_project_cache="Questions")
```

### Deleting Multiple Project Caches

To delete caches for multiple projects, use:

```python
llm.generate(request_list, project_name="New", clear_project_cache=["Maths", "Generate_data"])
```

### Deleting Entire Cache

To delete the entire cache for all projects, use:

```python
llm.generate(request_list, project_name="New", clear_project_cache="full")
```

## Conclusion

This documentation provides a comprehensive guide to using the `AsyncLLM` class for generating responses and managing caches effectively. By following the examples and guidelines, you can streamline your interactions with the language model while maintaining efficient cache management.
