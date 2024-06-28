# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
import typing

from dataformer.utils.notebook import in_notebook
if in_notebook():
    import nest_asyncio
    nest_asyncio.apply()


class OpenLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        api_provider: str = "openai",
        model: str = "",
        max_requests_per_minute: float = 20.0,
        max_tokens_per_minute: float = 5000.0,
        max_attempts: int = 5,
        token_encoding_name: str = "cl100k_base",
        logging_level: int = logging.INFO,
        gen_type: str = "chat"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_provider = api_provider.lower()
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_attempts = max_attempts
        self.token_encoding_name = token_encoding_name
        self.logging_level = logging_level
        self.gen_type = gen_type

        if model:
            self.model = model
        else:
            if self.api_provider == "openai" or "api.openai.com" in base_url:
                self.model = "gpt-3.5-turbo"
            elif self.api_provider == "groq":
                self.model = "mixtral-8x7b-32768"
            elif self.api_provider == "anthropic":
                self.model = "claude-2.1"
            elif self.api_provider == "together":
                self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            elif self.api_provider == "anyscale":
                self.model = "mistralai/Mistral-7B-Instruct-v0.1"
            elif self.api_provider == "deepinfra":
                self.model = "meta-llama/Meta-Llama-3-8B-Instruct"
            elif self.api_provider == "openrouter":
                self.model = "openai/gpt-3.5-turbo"
            else:
                raise ValueError("Specify the model you want to use.")

        logging.basicConfig(level=self.logging_level)
        self.logger = logging.getLogger(__name__)

    def get_request_url(self) -> str:
        if self.base_url:
            return self.base_url
        
        url_map = {
            "openai": {
                "chat": "https://api.openai.com/v1/chat/completions",
                "text": "https://api.openai.com/v1/completions"
            },
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "anthropic": {
                "chat": "https://api.anthropic.com/v1/messages",
                "text": "https://api.anthropic.com/v1/complete"
            },
            "together": "https://api.together.xyz/v1/chat/completions",
            "anyscale": "https://api.endpoints.anyscale.com/v1/chat/completions",
            "deepinfra": "https://api.deepinfra.com/v1/openai/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions"
        }
        
        if self.api_provider in url_map:
            if isinstance(url_map[self.api_provider], dict):
                if self.gen_type in url_map[self.api_provider]:
                    return url_map[self.api_provider][self.gen_type]
                else:
                    raise ValueError(f"Invalid gen_type provided for {self.api_provider}")
            else:
                return url_map[self.api_provider]
        else:
            raise ValueError(f"Invalid API Provider: {self.api_provider}")

    def get_api_key(self):
        if self.api_key:
            return self.api_key
        
        api_key_map = {
            "api.openai.com": "OPENAI_API_KEY",
            "api.groq.com": "GROQ_API_KEY",
            "api.anthropic.com": "ANTHROPIC_API_KEY",
            "api.together.xyz": "TOGETHER_API_KEY",
            "api.endpoints.anyscale.com": "ANYSCALE_API_KEY",
            "api.deepinfra.com": "DEEPINFRA_API_KEY",
            "openrouter.ai": "OPENROUTER_API_KEY"
        }
        
        for domain, env_var in api_key_map.items():
            if domain in self.base_url:
                api_key = os.getenv(env_var)
                if api_key:
                    return api_key
                else:
                    raise ValueError(f"API key not found in environment variable {env_var}")
        
        raise ValueError("Invalid API Key Provided")

    def get_requesturl_apikey(self) -> tuple[str, str]:
        request_url = self.get_request_url()
        api_key = self.get_api_key()
        return request_url, api_key

    async def process_api_requests(
        self,
        request_list: List[Dict[str, Any]],
        save_filepath: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        request_url, api_key = self.get_requesturl_apikey()

        seconds_to_pause_after_rate_limit_error = 15
        seconds_to_sleep_each_loop = 0.001

        api_endpoint = self.api_endpoint_from_url(request_url)
        request_header = {"Authorization": f"Bearer {api_key}"}
        if "/deployments" in request_url:  # for Azure OpenAI
            request_header = {"api-key": f"{api_key}"}
        elif self.api_provider == "anthropic":
            request_header = {"x-api-key": api_key, "content-type": "application/json"}

        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = self.task_id_generator_function()
        status_tracker = StatusTracker()
        next_request = None

        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        list_not_finished = True
        self.response_list = []

        requests = iter(request_list)
        self.logger.debug("Initialization complete. Entering main loop")
        
        async with aiohttp.ClientSession() as session:
            while True:
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        self.logger.debug(f"Retrying request {next_request.task_id}: {next_request}")
                    elif list_not_finished:
                        try:
                            request_json = next(requests)
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=self.num_tokens_consumed_from_request(request_json, api_endpoint),
                                attempts_left=self.max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            self.logger.debug(f"Reading request {next_request.task_id}: {next_request}")
                        except StopIteration:
                            self.logger.debug("Request list exhausted")
                            list_not_finished = False

                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
                last_update_time = current_time

                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                                openllm_instance=self
                            )
                        )
                        next_request = None

                if status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(seconds_to_sleep_each_loop)

                seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    await asyncio.sleep(remaining_seconds_to_pause)
                    self.logger.warning(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        if save_filepath:
            self.logger.info(f"Parallel processing complete. Results saved to {save_filepath}")
        if status_tracker.num_tasks_failed > 0:
            self.logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed.")
        if status_tracker.num_rate_limit_errors > 0:
            self.logger.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

        return sorted(self.response_list, key=lambda x: x[0]['idx'])

    def api_endpoint_from_url(self, request_url: str) -> str:
        match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
        if match is None:
            match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
        return match[1] if match else request_url

    def num_tokens_consumed_from_request(self, request_json: Dict[str, Any], api_endpoint: str) -> int:
        encoding = tiktoken.get_encoding(self.token_encoding_name)
        if api_endpoint.endswith("completions") or api_endpoint.endswith("messages") or "generateContent" in api_endpoint:
            max_tokens = request_json.get("max_tokens", 15)
            n = request_json.get("n", 1)
            completion_tokens = n * max_tokens

            if "messages" in request_json:
                num_tokens = 0
                for message in request_json["messages"]:
                    num_tokens += 4
                    for key, value in message.items():
                        num_tokens += len(encoding.encode(value))
                        if key == "name":
                            num_tokens -= 1
                num_tokens += 2
                return num_tokens + completion_tokens
            elif "prompt" in request_json:
                prompt = request_json["prompt"]
                if isinstance(prompt, str):
                    prompt_tokens = len(encoding.encode(prompt))
                    return prompt_tokens + completion_tokens
                elif isinstance(prompt, list):
                    prompt_tokens = sum(len(encoding.encode(p)) for p in prompt)
                    return prompt_tokens + completion_tokens * len(prompt)
                else:
                    raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
        elif api_endpoint == "embeddings":
            input_text = request_json["input"]
            if isinstance(input_text, str):
                return len(encoding.encode(input_text))
            elif isinstance(input_text, list):
                return sum(len(encoding.encode(i)) for i in input_text)
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
        else:
            raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')

    @staticmethod
    def task_id_generator_function():
        task_id = 0
        while True:
            yield task_id
            task_id += 1

    async def call_api(self, session: aiohttp.ClientSession, request_url: str, request_header: Dict[str, str], request_json: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_provider == "gemini":
            content = request_json["messages"][0]["content"] if "messages" in request_json else request_json.get("prompt", "")
            gemini_request = {
                "contents": [{"parts": [{"text": content}]}]
            }
            async with session.post(url=request_url, json=gemini_request) as response:
                response_json = await response.json()
                return {"choices": [{"message": {"content": response_json["candidates"][0]["content"]["parts"][0]["text"]}}]}
        elif self.api_provider == "anthropic":
            if self.gen_type == "chat":
                async with session.post(url=request_url, headers=request_header, json=request_json) as response:
                    response_json = await response.json()
                    return {"choices": [{"message": {"content": response_json["content"][0]["text"]}}]}
            elif self.gen_type == "text":
                async with session.post(url=request_url, headers=request_header, json=request_json) as response:
                    response_json = await response.json()
                    return {"choices": [{"text": response_json["completion"]}]}
        else:
            async with session.post(url=request_url, headers=request_header, json=request_json) as response:
                return await response.json()

    def generate(self, request_list: List[Dict[str, Any]], save_filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        for request in request_list:
            if "model" not in request:
                request["model"] = self.model

            if self.api_provider == "anthropic":
                if self.gen_type == "chat":
                    if "messages" in request:
                        request["messages"] = [{"role": msg["role"], "content": msg["content"]} for msg in request["messages"]]
                elif self.gen_type == "text":
                    if "prompt" in request:
                        request["prompt"] = request["prompt"]
            elif self.api_provider == "openrouter":
                request["model"] = self.model
        
        if save_filepath:
            with open(save_filepath, "w") as f:
                pass
        
        return asyncio.run(self.process_api_requests(request_list=request_list, save_filepath=save_filepath))


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest():
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
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
        save_filepath: str,
        status_tracker: StatusTracker,
        openllm_instance
    ):
        """Calls the OpenAI API and saves results."""
        
        
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1 
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [{'idx':self.task_id}, self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [{'idx':self.task_id}, self.request_json, [str(e) for e in self.result]]
                )
                if data is not None:
                    openllm_instance.response_list.append(data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [{'idx':self.task_id}, self.request_json, response, self.metadata]
                if self.metadata
                else [{'idx':self.task_id}, self.request_json, response]
            )
            if data is not None:
                openllm_instance.response_list.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
