# AsyncLLM Documentation

The `AsyncLLM` class is part of the `dataformer` library, designed to facilitate the use of asynchronous large language models (LLMs) in applications. This documentation provides an overview of the class, its parameters, methods, and associated components.

## Overview

The `AsyncLLM` class allows developers to generate text responses based on input prompts while managing various parameters that control the behavior and performance of the model. It supports concurrent API calls, rate limiting, and error handling.

## Class Definition

```python
class AsyncLLM:
    def __init__(
        self,
        api_provider="openai",
        model="",
        api_key=None,
        url="",        
        sampling_params={},
        max_requests_per_minute=None,
        max_tokens_per_minute=None,
        max_concurrent_requests=None,
        max_rps=False,
        max_attempts=3,
        token_encoding_name="cl100k_base",
        logging_level=logging.INFO,
        gen_type="chat",
        project_name=None,
        cache_dir=".cache/dataformer"
    ):
        # Initialization code
```

### Parameters

- **api_provider**: (str) Specifies the API provider for the LLM (e.g., "openai", "groq").
- **model**: (str) The specific model to use from the API provider.
- **api_key**: (str) Your API key for authentication with the API provider.
- **url**: (str) An optional URL for the API endpoint.
- **sampling_params**: (dict) A dictionary of parameters that control the sampling behavior of the model.
- **max_requests_per_minute**: (int) Limits the number of requests that can be made to the API per minute.
- **max_tokens_per_minute**: (int) Limits the number of tokens that can be processed per minute.
- **max_concurrent_requests**: (int) Specifies the maximum number of requests that can be processed concurrently.
- **max_rps**: (bool) A flag that, when set to `True`, limits the number of requests per second.
- **max_attempts**: (int) The number of retry attempts to make in case of a failure when generating responses.
- **token_encoding_name**: (str) Specifies the token encoding scheme to use.
- **logging_level**: (int) Sets the logging level for the application.
- **gen_type**: (str) Specifies the type of generation to use (e.g., "chat").
- **project_name**: (str) An optional name for the project.
- **cache_dir**: (str) The directory where cached data will be stored.

## Key Components

### StatusTracker Class

The `StatusTracker` class is used to store metadata about the script's progress. It tracks the number of tasks started, in progress, succeeded, failed, and any rate limit or API errors encountered.

```python
@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0
```

### APIRequest Class

The `APIRequest` class stores an API request's inputs, outputs, and other metadata. It contains a method to make an API call.

```python
@dataclass
class APIRequest:
    task_id: int
    recent_task_id: int
    difference_task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        cache_filepath: str,
        association_filepath: str,
        project_name: str,
        status_tracker: StatusTracker,
        asyncllm_instance,
    ):
        """Calls the OpenAI API and saves results."""
```

### Key Methods

- **check_model_exists**: Verifies if the specified model exists for the given API provider.
- **process_api_requests**: Handles the processing of API requests in parallel while adhering to rate limits.
- **generate**: Main method to generate responses based on the provided request list.

### Example Usage

Here’s an example of how to use the `AsyncLLM` class:

```python
from dataformer.llms import AsyncLLM

llm_params = {
    "api_provider": "groq",
    "model": "gpt-3.5-turbo",
    "api_key": "your_api_key_here",
    "url": "https://api.groq.com/v1",
    "sampling_params": {"temperature": 0.7},
    "max_requests_per_minute": 60,
    "max_tokens_per_minute": 1000,
    "max_concurrent_requests": 5,
    "max_rps": True,
    "max_attempts": 3,
    "token_encoding_name": "cl100k_base",
    "logging_level": "logging.INFO",
    "gen_type": "chat",
    "project_name": "MyProject",
    "cache_dir": "/path/to/cache"
}

llm = AsyncLLM(**llm_params)
response_list = llm.generate(request_list)
```

## Conclusion

The `AsyncLLM` class provides a robust framework for integrating asynchronous large language models into applications. By managing API requests efficiently and handling errors gracefully, it allows developers to focus on building features rather than dealing with the intricacies of API interactions.
