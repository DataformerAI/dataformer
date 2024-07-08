# imports
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit

# for storing API inputs, outputs, and metadata
import typing
from dataclasses import (
    dataclass,
    field,
)

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens

from dataformer.utils.cache import (
    create_hash,
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
)
from dataformer.llms.api_providers import model_dict, base_url_dict, api_key_dict
from dataformer.utils.notebook import in_notebook

if in_notebook():
    import nest_asyncio

    nest_asyncio.apply()


class OpenLLM:
    def __init__(
        self,
        api_key=None,
        base_url="",
        api_provider="openai",
        model="",
        max_requests_per_minute=None,
        max_tokens_per_minute=None,
        max_attempts=5,
        token_encoding_name="cl100k_base",
        logging_level=logging.INFO,
        gen_type="chat",
        cache_dir=".cache",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_provider = api_provider
        self.max_requests_per_minute = max_requests_per_minute or os.getenv('MAX_REQUESTS_PER_MINUTE', 20)
        self.max_tokens_per_minute = max_tokens_per_minute or os.getenv('MAX_TOKENS_PER_MINUTE', 10000)
        self.max_attempts = max_attempts
        self.token_encoding_name = token_encoding_name
        self.logging_level = logging_level
        self.gen_type = gen_type
        self.skip_task_ids = []
        self.cache_dir = cache_dir
        self.task_id_generator = None

        # initialize logging
        logging.basicConfig(level=self.logging_level, force=True)

        if model:
            self.model = model
        elif self.base_url or self.api_provider:
            self.base_url = self.base_url or self.get_request_url()
            self.model = model_dict.get(self.base_url)
        else:
            raise ValueError("Model not provided.")

    def get_request_url(self):
        if self.base_url:
            return self.base_url

        if self.api_provider in base_url_dict:
            if isinstance(base_url_dict[self.api_provider], dict):
                if self.gen_type in base_url_dict[self.api_provider]:
                    self.base_url = base_url_dict[self.api_provider][self.gen_type]
                else:
                    raise ValueError("Invalid gen_type provided")
            else:
                self.base_url = base_url_dict[self.api_provider]
        else:
            raise ValueError("Invalid API Provider")

        return self.base_url

    def get_api_key(self):
        if self.api_key:
            return self.api_key
        else:
            for base_url, env_var in api_key_dict.items():
                if base_url in self.base_url:
                    return os.getenv(env_var)

        raise ValueError("Invalid API Key Provided")

    def get_requesturl_apikey(self):
        request_url = self.get_request_url()
        api_key = self.get_api_key()
        return request_url, api_key

    async def process_api_requests(
        self, request_url: str, api_key: str, request_list: list, cache_filepath: str
    ):
        """Processes API requests in parallel, throttling to stay under rate limits."""

        # constants
        seconds_to_pause_after_rate_limit_error = 15
        seconds_to_sleep_each_loop = (
            0.001  # 1 ms limits max throughput to 1,000 requests per second
        )

        logging.debug(f"Logging initialized at level {self.logging_level}")

        # infer API endpoint and construct request header
        api_endpoint = self.api_endpoint_from_url(request_url)
        request_header = {"Authorization": f"Bearer {api_key}"}
        # use api-key header for Azure deployments
        if "/deployments" in request_url:
            request_header = {"api-key": f"{api_key}"}

        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        if not self.task_id_generator:
            self.task_id_generator = (
                task_id_generator_function()
            )  # generates integer IDs of 0, 1, 2, ...
        status_tracker = (
            StatusTracker()
        )  # single instance to track a collection of variables
        next_request = None  # variable to hold the next request to call

        # initialize available capacity counts
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        # initialize flags
        list_not_finished = True  # after list is empty, we'll skip reading it
        logging.debug("Initialization complete.")

        # initialize list reading
        requests = iter(request_list)
        logging.debug("List opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif list_not_finished:
                        try:
                            # get new request
                            request_json = next(requests)
                            active_task_id = next(self.task_id_generator)
                            if active_task_id in self.skip_task_ids:
                                logging.info(
                                    f"[Cache Used] Skip request  {str(active_task_id)}"
                                )
                                continue
                            next_request = APIRequest(
                                task_id=active_task_id,
                                request_json=request_json,
                                token_consumption=self.num_tokens_consumed_from_request(
                                    request_json, api_endpoint, self.token_encoding_name
                                ),
                                attempts_left=self.max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if list runs out, set flag to stop reading it
                            logging.debug("Read list exhausted")
                            list_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                cache_filepath=cache_filepath,
                                status_tracker=status_tracker,
                                openllm_instance=self,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

    def api_endpoint_from_url(self, request_url):
        """Extract the API endpoint from the request URL."""
        match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
        if match is None:
            # for Azure OpenAI deployment urls
            match = re.search(
                r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
            )

        if match is None:
            return request_url
        else:
            return match[1]

    def num_tokens_consumed_from_request(
        self,
        request_json: dict,
        api_endpoint: str,
        token_encoding_name: str,
    ):
        """Count the number of tokens in the request. Only supports completion and embedding requests."""
        encoding = tiktoken.get_encoding(token_encoding_name)
        # if completions request, tokens = prompt + n * max_tokens
        if api_endpoint.endswith("completions"):
            max_tokens = request_json.get("max_tokens", 15)
            n = request_json.get("n", 1)
            completion_tokens = n * max_tokens

            # chat completions
            # if api_endpoint.startswith("chat/"):
            if "chat/" in api_endpoint:
                num_tokens = 0
                for message in request_json["messages"]:
                    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                        num_tokens += len(encoding.encode(value))
                        if key == "name":  # if there's a name, the role is omitted
                            num_tokens -= (
                                1  # role is always required and always 1 token
                            )
                num_tokens += 2  # every reply is primed with <im_start>assistant
                return num_tokens + completion_tokens
            # normal completions
            else:
                prompt = request_json["prompt"]
                if isinstance(prompt, str):  # single prompt
                    prompt_tokens = len(encoding.encode(prompt))
                    num_tokens = prompt_tokens + completion_tokens
                    return num_tokens
                elif isinstance(prompt, list):  # multiple prompts
                    prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                    num_tokens = prompt_tokens + completion_tokens * len(prompt)
                    return num_tokens
                else:
                    raise TypeError(
                        'Expecting either string or list of strings for "prompt" field in completion request'
                    )
        # if embeddings request, tokens = input tokens
        elif api_endpoint == "embeddings":
            input = request_json["input"]
            if isinstance(input, str):  # single input
                num_tokens = len(encoding.encode(input))
                return num_tokens
            elif isinstance(input, list):  # multiple inputs
                num_tokens = sum([len(encoding.encode(i)) for i in input])
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "inputs" field in embedding request'
                )
        # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
        else:
            raise NotImplementedError(
                f'API endpoint "{api_endpoint}" not implemented in this script'
            )

    def generate(
        self,
        request_list: typing.List,
        cache_vars: typing.Dict = {},
        task_id_generator=None,
        use_cache=True,
    ):
        # Set base_url before any caching
        request_url, api_key = self.get_requesturl_apikey()

        if task_id_generator:
            self.task_id_generator = task_id_generator

        self.response_list = []

        # If cache, add cached responses to self.response_list
        if "request_details" not in cache_vars:
            self.request_details = get_request_details(request_list)

        cache_vars = get_cache_vars(
            self,
            ignore_keys=[
                "cache_hash",
                "skip_task_ids",
                "task_id_generator",
                "response_list",
                "use_cache",
            ],
            additional_data=cache_vars,
        )
        # We create cache_hash to sort the async respones even when use_cache is False.
        self.cache_hash = create_hash(cache_vars)

        self.skip_task_ids = []
        cache_filepath = None
        if use_cache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_filepath = f"{self.cache_dir}/{str(self.cache_hash)}.jsonl"

            if os.path.exists(cache_filepath):
                with open(cache_filepath, "r") as f:
                    cached_responses = f.readlines()
                cached_indices = []
                for response in cached_responses:
                    response_json = json.loads(response)
                    self.response_list.append(response_json)
                    if self.cache_hash in response_json[0].keys():
                        cached_indices.append(response_json[0][self.cache_hash])
                self.skip_task_ids = cached_indices

        for request in request_list:
            request["model"] = self.model

        asyncio.run(
            self.process_api_requests(
                request_url,
                api_key,
                request_list=request_list,
                cache_filepath=cache_filepath,
            )
        )

        sorted_response_list = sorted(
            self.response_list, key=lambda x: x[0][self.cache_hash]
        )
        sorted_response_list = [
            item[1:] for item in sorted_response_list
        ]  # Exclude self.cache_hash from the list

        return sorted_response_list


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
class APIRequest:
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
        cache_filepath: str,
        status_tracker: StatusTracker,
        openllm_instance,
    ):
        """Calls the OpenAI API and saves results."""

        if self.task_id in openllm_instance.skip_task_ids:
            return  # Skip request

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
                        1  # rate limit errors are counted separately
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
                    [
                        {openllm_instance.cache_hash: self.task_id},
                        self.request_json,
                        [str(e) for e in self.result],
                        self.metadata,
                    ]
                    if self.metadata
                    else [
                        {openllm_instance.cache_hash: self.task_id},
                        self.request_json,
                        [str(e) for e in self.result],
                    ]
                )
                if data is not None:
                    openllm_instance.response_list.append(data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [
                    {openllm_instance.cache_hash: self.task_id},
                    self.request_json,
                    response,
                    self.metadata,
                ]
                if self.metadata
                else [
                    {openllm_instance.cache_hash: self.task_id},
                    self.request_json,
                    response,
                ]
            )
            if data is not None:
                openllm_instance.response_list.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

            # Save the response immediately after processing the request
            if cache_filepath is not None:
                data = (
                    [
                        {openllm_instance.cache_hash: self.task_id},
                        self.request_json,
                        response,
                        self.metadata,
                    ]
                    if self.metadata
                    else [
                        {openllm_instance.cache_hash: self.task_id},
                        self.request_json,
                        response,
                    ]
                )
                if data is not None:
                    json_string = json.dumps(data)
                    with open(cache_filepath, "a") as f:
                        f.write(json_string + "\n")
