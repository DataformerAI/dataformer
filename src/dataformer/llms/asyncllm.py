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

from dataformer.llms.api_providers import api_key_dict, base_url_dict, model_dict
from dataformer.utils.cache import (
    create_hash,
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
)
from dataformer.utils.notebook import in_notebook

if in_notebook():
    import nest_asyncio

    nest_asyncio.apply()


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
        asyncllm_instance,
    ):
        """Calls the OpenAI API and saves results."""

        if self.task_id in asyncllm_instance.skip_task_ids:
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

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
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
                response = [str(e) for e in self.result]
                data = self.create_data(asyncllm_instance, response)
                if data is not None:
                    asyncllm_instance.response_list.append(data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            response = self.convert_response(request_url, response)
            data = self.create_data(asyncllm_instance, response)

            if data is not None:
                asyncllm_instance.response_list.append(data)
                # Save the response immediately after processing the request
                if cache_filepath is not None:
                    json_string = json.dumps(data)
                    with open(cache_filepath, "a") as f:
                        f.write(json_string + "\n")

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

    def create_data(self, asyncllm_instance, response):
        data = [
            {asyncllm_instance.cache_hash: self.task_id},
            self.request_json,
            response,
        ] + ([self.metadata] if self.metadata else [])
        return data

    def convert_response(self, request_url, response):
        ollama_keys = ("ollama", "11434", "api/chat", "api/generate")
        if request_url.endswith("completions"):
            return response
        elif "anthropic" in request_url:
            response["usage"] = {
                "prompt_tokens": response["usage"]["input_tokens"],
                "completion_tokens": response["usage"]["output_tokens"],
                "total_tokens": response["usage"]["input_tokens"]
                + response["usage"]["output_tokens"],
            }
            response["object"] = response.pop("type", "")
            response["choices"] = [
                {
                    "index": response.pop("stop_sequence", ""),
                    "message": {
                        "role": response.pop("role", ""),
                        "content": response.pop("content")[0]["text"],
                    },
                    "finish_reason": response.pop("stop_reason", ""),
                }
            ]
            response["created"] = int(time.time())
            return response
        elif any(word in request_url for word in ollama_keys):
            response["object"] = (
                "ollama.chat" if "chat" in request_url else "ollama.generate"
            )
            response["created"] = response.pop("created_at", int(time.time()))
            response["system_fingerprint"] = "fp_ollama"
            response["choices"] = [
                {
                    "index": 0,
                    "message": response.pop("message", ""),
                    "finish_reason": response.pop("done_reason", ""),
                }
            ]
            response["usage"] = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.pop("prompt_eval_count", 0)
                + response.pop("eval_count", 0),
            }

            for key in (
                "done",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            ):
                response.pop(key, "")
            return response
        else:
            return response


class AsyncLLM:
    def __init__(
        self,
        api_key=None,
        base_url="",
        api_provider="openai",
        model="",
        sampling_params={},
        max_requests_per_minute=None,
        max_tokens_per_minute=None,
        max_concurrent_requests=None,
        max_rps=False,
        max_attempts=5,
        token_encoding_name="cl100k_base",
        logging_level=logging.INFO,
        gen_type="chat",
        cache_dir=".cache/dataformer",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_provider = api_provider
        self.max_requests_per_minute = max_requests_per_minute or os.getenv(
            "MAX_REQUESTS_PER_MINUTE", 60
        )
        self.max_tokens_per_minute = max_tokens_per_minute or os.getenv(
            "MAX_TOKENS_PER_MINUTE", 10000000000
        )
        self.max_concurrent_requests = max_concurrent_requests or os.getenv(
            "MAX_TOKENS_PER_MINUTE"
        )
        self.max_rps = max_rps
        self.max_attempts = max_attempts
        self.token_encoding_name = token_encoding_name
        self.logging_level = logging_level
        self.gen_type = gen_type
        self.skip_task_ids = []
        self.cache_dir = cache_dir
        self.sampling_params = sampling_params
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

        if self.api_provider == "together":
            self.max_rps = True

    def get_request_url(self):
        if self.base_url:
            if not self.api_provider:
                # If api_provider is not set, get it from base_url_dict
                for provider, urls in base_url_dict.items():
                    if self.base_url in urls.values():
                        self.api_provider = provider
                        break

            if self.api_provider == "together":
                self.max_rps = True

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
   
    #def get_request_url_from_values(self,base_url,api_provider,gen_type="chat"):
        
    def get_api_key(self):
        if self.api_provider == "ollama":
            return "ollama"
        if self.api_key:
            return self.api_key
        else:
            for base_url, env_var in api_key_dict.items():
                if base_url in self.base_url:
                    #send key only if it exists
                    if os.getenv(env_var):
                        return os.getenv(env_var)

        raise ValueError("Invalid API Key Provided")

    def get_requesturl_apikey(self):
        request_url = self.get_request_url()
        api_key = self.get_api_key()
        return request_url, api_key
    #Send request url and api provider
    def get_requesturl_apikey_list(self,base_url,api_provider):
        gen_type = self.gen_type
        if base_url:
            if not api_provider:
                # If api_provider is not set, get it from base_url_dict
                for provider, urls in base_url_dict.items():
                    if base_url in urls.values():
                        api_provider = provider
                        break

            

            return [base_url,api_provider]

        if api_provider in base_url_dict:
            if isinstance(base_url_dict[api_provider], dict):
                if gen_type in base_url_dict[api_provider]:
                    base_url = base_url_dict[api_provider][gen_type]
                else:
                    raise ValueError("Invalid gen_type provided")
            else:
                base_url = base_url_dict[api_provider]
        else:
            raise ValueError("Invalid API Provider")

        return [base_url,api_provider]
    

    async def process_api_requests(
        self, request_url: str, api_key: str, request_list: list, cache_filepath: str,different_requests=False
    ):
        """Processes API requests in parallel, throttling to stay under rate limits."""

        # constants
        seconds_to_pause_after_rate_limit_error = 15
        seconds_to_sleep_each_loop = (
            0.001  # 1 ms limits max throughput to 1,000 requests per second
        )

        logging.debug(f"Logging initialized at level {self.logging_level}")

        if not different_requests:
                # infer API endpoint and construct request header
            api_endpoint = self.api_endpoint_from_url(request_url)
            request_header = {"Authorization": f"Bearer {api_key}"}
            # use api-key header for Azure deployments
            if "/deployments" in request_url:
                request_header = {"api-key": f"{api_key}"}
            # Use x-api-key for Anthropic
            if self.api_provider == "anthropic":
                request_header = {
                    "x-api-key": f"{api_key}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
            if self.api_provider == "ollama":
                request_header = {
                    "Content-Type": "application/json",
                    "api_key": api_key,
                }
        else:
            #first request then set self.max_rps to True if provider is together
            if request_list[0]["api_provider"]=="together":
                self.max_rps=True

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
        available_request_capacity = (
            self.max_requests_per_minute / 60
            if self.max_rps
            else self.max_requests_per_minute
        )
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        # initialize flags
        list_not_finished = True  # after list is empty, we'll skip reading it
        logging.debug("Initialization complete.")

        semaphore = (
            asyncio.Semaphore(self.max_concurrent_requests)
            if self.max_concurrent_requests
            else None
        )

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
                            if different_requests:
                                    ##infer API endpoint and construct request header
                                    request_url = request_json["base_url"]
                                    api_key = request_json["api_key"]
                                    api_provider = request_json["api_provider"]
                                    if api_provider=="together":
                                        self.max_rps=True
                                    else:
                                        self.max_rps=False
                                        
                                    api_endpoint = self.api_endpoint_from_url(request_url)
                                    request_header = {"Authorization": f"Bearer {api_key}"}
                                    # use api-key header for Azure deployments
                                    if "/deployments" in request_url:
                                        request_header = {"api-key": f"{api_key}"}
                                    # Use x-api-key for Anthropic
                                    if api_provider == "anthropic":
                                        request_header = {
                                            "x-api-key": f"{api_key}",
                                            "anthropic-version": "2023-06-01",
                                            "content-type": "application/json",
                                        }
                                    if api_provider == "ollama":
                                        request_header = {
                                            "Content-Type": "application/json",
                                            "api_key": api_key,
                                        }
                            #delete the extra keys used above
                            del request_json["api_key"]
                            del request_json["base_url"]
                            del request_json["api_provider"]
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
                if self.max_rps:
                    # If max_rps is True, divide max_requests_per_minute by 60
                    available_request_capacity = min(
                        available_request_capacity
                        + (self.max_requests_per_minute / 60) * seconds_since_update,
                        self.max_requests_per_minute / 60,
                    )
                else:
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
                    if self.max_concurrent_requests is not None or (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # call API
                        if semaphore:
                            asyncio.create_task(
                                self._call_api_with_semaphore(
                                    semaphore,
                                    next_request,
                                    session,
                                    request_url,
                                    request_header,
                                    queue_of_requests_to_retry,
                                    cache_filepath,
                                    status_tracker,
                                )
                            )
                        else:
                            # update counters
                            available_request_capacity -= 1
                            available_token_capacity -= next_request_tokens
                            next_request.attempts_left -= 1

                            asyncio.create_task(
                                next_request.call_api(
                                    session=session,
                                    request_url=request_url,
                                    request_header=request_header,
                                    retry_queue=queue_of_requests_to_retry,
                                    cache_filepath=cache_filepath,
                                    status_tracker=status_tracker,
                                    asyncllm_instance=self,
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

        ollama_flag = any(
            word in self.base_url
            for word in ("ollama", "11434", "api/chat", "api/generate")
        )

        # if completions request, tokens = prompt + n * max_tokens
        if api_endpoint.endswith(("completions", "messages")) or ollama_flag:
            max_tokens = request_json.get("max_tokens", 15)
            n = request_json.get("n", 1)
            completion_tokens = n * max_tokens

            # chat completions
            # if api_endpoint.startswith("chat/"):
            if "chat/" in api_endpoint or "messages" in api_endpoint or ollama_flag:
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
        elif api_endpoint.endswith("embeddings"):
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

    async def _call_api_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        request: APIRequest,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        cache_filepath: str,
        status_tracker: StatusTracker,
    ):
        async with semaphore:
            await request.call_api(
                session=session,
                request_url=request_url,
                request_header=request_header,
                retry_queue=retry_queue,
                cache_filepath=cache_filepath,
                status_tracker=status_tracker,
                asyncllm_instance=self,
            )

    def generate(
        self,
        request_list: typing.List,
        cache_vars: typing.Dict = {},
        task_id_generator=None,
        use_cache=True,
        clear_prev_cache=False,
    ):
          
                       
        #keys to ignore            
        ignore_keys = [
            "cache_hash",
            "skip_task_ids",
            "task_id_generator",
            "response_list",
            "use_cache",
            "max_requests_per_minute",
            "max_tokens_per_minute",
            "max_attempts",
            "max_concurrent_requests",
            "max_rps",
            "api_key",
            "sampling_params" # Already part of request_list
        ]

        # Check if 'model' is present in all request items
        if all("model" in request for request in request_list):
            # Since all request already have model, we can ignore self.model for cache.
            ignore_keys.append("model")
        else:
            for request in request_list:
                #over ride model based on the api provider or base url in the request list
                if "model" not in request:
                    model=self.model
                    api_provider=""
                    base_url=""
                    if "base_url" in list(request.keys()) or "api_provider" in list(request.keys()):
                        if "api_provider" in list(request.keys()):
                            api_provider=request['api_provider']
                        if "base_url" in list(request.keys()):
                            base_url=request['base_url']
                        base_url = self.get_requesturl_apikey_list(base_url,api_provider)[0]
                        
                        model = model_dict.get(base_url)
                    
                    request["model"] = model
                    


        request_url=""
        api_key=""
        different_apis=False
        #check if any provider or base url present in request list
        if any("api_provider" in request or "base_url" in request for request in request_list):
                different_apis = True
        
        #If no urls or api providers in request means use default in init
        if not different_apis:
            request_url = self.get_request_url()
            for requests in request_list:
                requests['api_provider'] = self.api_provider
                requests['base_url'] = request_url
                api_key=""
                if "api_key" in list(requests.keys()):
                    api_key=requests['api_key']
                else:
                    api_key = self.get_api_key()
                requests['api_key'] = api_key
        else:
                  
            if all("api_provider" in request or "base_url" in request for request in request_list):
                # Since all request already have either api_provider or base_url, we can ignore self.provider or if base_url (if set) for cache.
                # all requests have either one of api_provider or base_url then no need for initatied or default values for cache hash calculation
                # #because they actuallhy wont be used in requests
                
                    ignore_keys.append("api_provider")
                    ignore_keys.append("base_url")
               
            else:
                
                use_default_or_request_api=[]
                for request in request_list:
                    key_list = list(request.keys())
                    if "api_provider" in key_list or "base_url" in key_list:
                       
                        #Use another provider than default or in llm init
                        use_default_or_request_api.append(False) 
                    else:
                        #Use same provider as default or in llm init
                        use_default_or_request_api.append(True)

        
                #if default init is used, check if api key given in request list instead of init
                for i in range(len(request_list)):
                    api_key=""
                    if use_default_or_request_api[i]:
                        if "api_key" in list(request_list[i].keys()):
                            api_key = request_list[i]['api_key']
                        
                        if api_key!="":
                            base_url = self.get_request_url()
                        else:
                            base_url,api_key = self.get_requesturl_apikey()
                        
                        api_provider = self.api_provider
                        request_list[i]['api_provider'] = api_provider
                        request_list[i]['base_url'] = base_url
                        request_list[i]['api_key'] = api_key
            
            #to skip records, already processed
            skip=True
            for i in range(len(request_list)):
                    api_provider=""
                    base_url=""
                    api_key=""
                    keys_list=list(request_list[i].keys())
                    #check if any field is missing
                    if "api_provider" not in keys_list or "api_key" not in keys_list or "base_url" not in keys_list:
                        skip=False
                        if "api_provider" in keys_list:
                            api_provider=request_list[i]['api_provider']
                        if "base_url" in keys_list:
                            base_url=request_list[i]['base_url']
                        #get the base url
                        base_url, api_provider = self.get_requesturl_apikey_list(base_url,api_provider)
                        
                        #if both not found in request list, may be in initialized init
                        if base_url=="" and api_provider=="":
                            base_url,api_key = self.get_requesturl_apikey()
                            api_provider = self.api_provider
                       #set the api key
                        if "api_key" in keys_list:
                            #Use another provider than default or in llm init
                            api_key= request_list[i]['api_key']
                        elif self.api_key:
                            api_key=self.api_key
                        elif api_provider=="ollama":
                            api_key="ollama"
                        else:
                            for base_url_var, env_var in api_key_dict.items():
                                if base_url_var in base_url:
                                    api_key  = os.getenv(env_var)
                                    break
                            

                        
                        if api_key=="" or api_key==None:
                            raise("Api key not provided or INVALID. Provide appropriate API key with api provider or baseurl.")
                    else:
                        skip=True
                        
                    if not skip:
                        if api_provider=="" or base_url=="":
                            raise("No api provider or base url provided")
                        else:
                            request_list[i]['api_provider'] = api_provider
                            request_list[i]['base_url'] = base_url
                            request_list[i]['api_key'] = api_key
                   

        for request in request_list:
            for key, value in self.sampling_params.items():
                if key not in request:
                    request[key] = value

        if task_id_generator:
            self.task_id_generator = task_id_generator

        self.response_list = []

        # If cache, add cached responses to self.response_list
        if "request_details" not in cache_vars:
            self.request_details = get_request_details(request_list)

        cache_vars = get_cache_vars(
            self,
            ignore_keys=ignore_keys,
            additional_data=cache_vars,
        )
       
        # We create cache_hash to sort the async responses even when use_cache is False.
        self.cache_hash = create_hash(cache_vars)

        self.skip_task_ids = []

        cache_filepath = f"{self.cache_dir}/{str(self.cache_hash)}.jsonl"

        if clear_prev_cache and os.path.exists(cache_filepath):
            os.remove(cache_filepath)

        if use_cache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

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
        else:
            cache_filepath = None
        
        if different_apis:
            #for different api providers
            asyncio.run(
                self.process_api_requests(
                    request_url,
                    api_key="",
                    request_list=request_list,
                    cache_filepath=cache_filepath,
                    different_requests=True
                )
            )
        else:
             #for same api provider
             asyncio.run(
                self.process_api_requests(
                    request_url,
                    api_key,
                    request_list=request_list,
                    cache_filepath=cache_filepath,
                    different_requests=False
                )
            )           

        sorted_response_list = sorted(
            self.response_list, key=lambda x: x[0][self.cache_hash]
        )
        sorted_response_list = [
            item[1:] for item in sorted_response_list
        ]  # Exclude self.cache_hash from the list

        return sorted_response_list