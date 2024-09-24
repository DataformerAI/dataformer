# imports
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit
import subprocess # for checking if the provided model really exists
import sys # for checking if the provided model really exists
# for storing API inputs, outputs, and metadata
import typing
from dataclasses import (
    dataclass,
    field,
)

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens

from dataformer.llms.api_providers import api_key_dict, url_dict, model_dict
from dataformer.utils.cache import (
    create_hash,
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
    delete_cache
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
    recent_task_id:int
    difference_task_id:int
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
        association_filepath:str,
        project_name:str,
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
            self.attempts_left -= 1
            logging.info(f"Request ID: {self.task_id}, Attempts Left: {self.attempts_left}")
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
                
                if association_filepath is not None:
                    self.update_association_file(association_filepath,cache_filepath,project_name)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

    def create_data(self, asyncllm_instance, response):
        data = [
            {asyncllm_instance.cache_hash[self.task_id-self.recent_task_id-self.difference_task_id]: self.task_id-self.recent_task_id-self.difference_task_id},
            self.request_json,
            response,
        ] + ([self.metadata] if self.metadata else [])
        return data
        
    def update_association_file(self,association_filepath,cache_filepath,project_name):
        with open(association_filepath,"r") as f:
            data = json.load(f)
        hash_data=[]
        #Iterate data and get hash
        if os.path.exists(cache_filepath):
            with open(cache_filepath,"r") as f:
                file = f.readlines()
                for row in file:
                    row = json.loads(row)
                    hash_data.append(list(row[0].keys())[0])
            
            filename = os.path.basename(cache_filepath).split(".")[0]
            dict_data_projects={}
            with open(association_filepath,"r") as file:
                json_data = json.load(file)
                dict_data_projects['project_requests'] = json_data['project_requests']
                #check if project name exists and append r create new association
                if project_name in list(json_data["project_requests"].keys()):
                    if filename not in dict_data_projects['project_requests'][project_name]:
                        dict_data_projects['project_requests'][project_name].append(filename)
                else:
                    dict_data_projects['project_requests'][project_name]=[filename]
               
            data[filename]= hash_data
            if "project_requests" in list(dict_data_projects.keys()):
                data["project_requests"] = dict_data_projects['project_requests']
            else:
                raise ValueError("Association file has incorrect associations.")
            with open(association_filepath,"w") as f:
                json.dump(data,f)
    
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
        self.api_key = api_key
        self.url = url
        self.api_provider = api_provider
        self.max_requests_per_minute = max_requests_per_minute or os.getenv(
            "MAX_REQUESTS_PER_MINUTE", 60
        )
        self.max_requests_per_minute = int(self.max_requests_per_minute)
        
        self.max_tokens_per_minute = max_tokens_per_minute or os.getenv(
            "MAX_TOKENS_PER_MINUTE", 10000000000
        )
        self.max_tokens_per_minute = int(self.max_tokens_per_minute)
        self.max_concurrent_requests = max_concurrent_requests or os.getenv(
            "MAX_TOKENS_PER_MINUTE"
        )
        if self.max_concurrent_requests:
            self.max_concurrent_requests = int(self.max_concurrent_requests)

        self.max_rps = max_rps
        self.max_attempts = max_attempts
        self.token_encoding_name = token_encoding_name
        self.logging_level = logging_level
        self.gen_type = gen_type
        self.skip_task_ids = []
        self.cache_dir = cache_dir
        self.sampling_params = sampling_params
        self.task_id_generator = None
        self.project_name = project_name or os.getenv(
            "PROJECT_NAME"
        )
        if self.project_name=="" or self.project_name is None:
            self.project_name= "dataformer"
        self.project_name=self.project_name.lower()
        # initialize logging
        logging.basicConfig(level=self.logging_level, force=True)

        if model:
            self.model = model
        elif self.url or self.api_provider:
            self.url = self.url or self.get_request_url()
            self.model = model_dict.get(self.url)
        else:
            raise ValueError("Model not provided.")
        if self.url=="":
            self.url = self.get_request_url()
        # Skip check_model_exists if URL is provided
        if not url:
            self.check_model_exists(self.url, self.api_provider, self.api_key, self.model)
        if self.api_provider == "together":
            self.max_rps = True

    def check_model_exists(self,api_url,api_provider,api_key,model):

        # Check if the api_url is in the url_dict
        if not any(api_url in urls.values() for urls in url_dict.values()):
            print(f"Skipping model verification for URL: {api_url}")
            return
        
        # check if the url and api_key exists
        # check if the api_key and model exists
        if api_key=="" or not api_key:
            api_key=self.get_api_key()
            if not api_key and not model:
                raise ValueError("API key not provided")
        # assign the api provider and url
        if api_provider=="" or not api_provider:
    
            for provider, urls in url_dict.items():
                if api_url in urls.values():
                    api_provider = provider
                    break
            if api_provider=="" or not api_provider:
                raise ValueError("No api provider found")
        
        # if api_provider api_url and model given go ahead
        if api_provider and model:
            
            #Get the url for making request to find out the models supported by the api 
            if isinstance(url_dict[api_provider], dict):
                    if "models" in url_dict[api_provider]:
                        url = url_dict[api_provider]["models"]
                    else:
                        url = url_dict[api_provider]["chat"] #for antropic only
            else:
                    url = url_dict[self.api_provider]
            
            # Make the GET request to get the models list  
            # anthropic doesn't have any list models api endpoint,check with message reply or error if any, to see if a amdoel exists or not
           
            if api_provider=="anthropic":
                data = {
                    "model": model,
                    "max_tokens": self.max_tokens_per_minute,
                    "messages": [
                        {
                            "role": "user", 
                            "content": "Hello, world"
                            }
                        ]
                }

                json_data = json.dumps(data)
                curl_command = [
                    "curl", "-s", "-X", "POST", url,
                    "--header", f"x-api-key: {api_key}",
                    "--header", "anthropic-version: 2023-06-01",
                    "--header", "Content-Type: application/json",
                    "--data", json_data
                ]
            
            else:
                curl_command = [
                    "curl",
                    "-s",
                    url,
                    "-H",
                    f"Authorization: Bearer {api_key}"
                ]  
            if sys.platform == "win32":
                CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW # For windows platforms
            else:
                CREATE_NO_WINDOW = 0  # For non-Windows platforms

            #execute the curl request and load the output in json format            
            try:
                output = subprocess.check_output(curl_command, text=True,creationflags=CREATE_NO_WINDOW,encoding="utf-8")
            except Exception:
                raise ValueError('Some exception occurred while testing for model support')
           
            try:
                response = json.loads(output)
            except Exception:
                raise ValueError("Tried to verify the model but received the following response from api provider.",output)
            
            # Convert response to dict if it's a list
            if isinstance(response, list):
                response = {"data": response}

            
            if "error" in list(response.keys()):
                 raise ValueError("Tried to verify the model but received the error from the api provider",response)
            elif api_provider=="anthropic":
                print("Model verified successfully")
                return
                     
            # if proper response received go for model checking
            if 'id' in list(response.keys()) or 'data' in list(response.keys()):
                    response = response['data']
                    models = [i['id'] for i in response]
                    #Check if the model exists/supported by the api provider platform
                    if not model in models:
                        raise ValueError("Wrong model provided for the api_provider or url. The model doesn't exist on the given api_provider or url.")
                    print("Model verified successfully")
            else:
                raise ValueError("tried to verify the model but received error." ,response)
            
    def get_request_url(self):
        if self.url:
            if not self.api_provider:
                # If api_provider is not set, get it from url_dict
                for provider, urls in url_dict.items():
                    if self.url in urls.values():
                        self.api_provider = provider
                        break

            if self.api_provider == "together":
                self.max_rps = True

            return self.url

        if self.api_provider in url_dict:
            if isinstance(url_dict[self.api_provider], dict):
                if self.gen_type in url_dict[self.api_provider]:
                    self.url = url_dict[self.api_provider][self.gen_type]
                else:
                    raise ValueError("Invalid gen_type provided")
            else:
                self.url = url_dict[self.api_provider]
        else:
            raise ValueError("Invalid API Provider")

        return self.url
   
       
    def get_api_key(self):
        if self.api_provider == "ollama":
            return "ollama"
        if self.api_key:
            return self.api_key
        else:
            for url, env_var in api_key_dict.items():
                if url in self.url:
                    #send key only if it exists
                    if os.getenv(env_var):
                        return os.getenv(env_var)

        raise ValueError("Invalid API Key Provided")

    def get_requesturl_apikey(self):
        request_url = self.get_request_url()
        api_key = self.get_api_key()
        return request_url, api_key
    #Send request url and api provider
    def get_requesturl_apikey_list(self,url,api_provider):
        gen_type = self.gen_type
        if url:
            if not api_provider:
                # If api_provider is not set, get it from url_dict
                for provider, urls in url_dict.items():
                    if url in urls.values():
                        api_provider = provider
                        break

            

            return [url,api_provider]

        if api_provider in url_dict:
            if isinstance(url_dict[api_provider], dict):
                if gen_type in url_dict[api_provider]:
                    url = url_dict[api_provider][gen_type]
                else:
                    raise ValueError("Invalid gen_type provided")
            else:
                url = url_dict[api_provider]
        else:
            raise ValueError("Invalid API Provider")

        return [url,api_provider]
    

    async def process_api_requests(
        self, request_url: str, api_key: str, request_list: list, cache_filepath: str, association_filepath:str,project_name:str, different_requests=False
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
            url = request_url
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
                for request in request_list:
                    request['max_tokens'] = self.max_tokens_per_minute
            if self.api_provider == "ollama":
                request_header = {
                    "Content-Type": "application/json",
                    "api_key": api_key,
                }
        else:
            url = request_list[0]["url"]
            #first request then set self.max_rps to True if provider is together
            if request_list[0]["api_provider"]=="together":
                self.max_rps=True
        first_request=True
        second_request=False
        task_id_recent=0
        value_iterate=1
        difference_task_id=0
        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        if self.task_id_generator is None:
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
        # Create a TCPConnector with SSL verification disabled for monsterapi
        connector = aiohttp.TCPConnector(ssl=False) if "monsterapi.ai" in url else None
        async with aiohttp.ClientSession(connector=connector) as session:  # Initialize ClientSession here
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
                                    request_url = request_json["url"]
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
                                        request_json['max_tokens'] = self.max_tokens_per_minute
                                    if api_provider == "ollama":
                                        request_header = {
                                            "Content-Type": "application/json",
                                            "api_key": api_key,
                                        }
                            #delete the extra keys used above
                            del request_json["api_key"]
                            del request_json["url"]
                            del request_json["api_provider"]
                            active_task_id = next(self.task_id_generator)
                            if second_request:
                                #when iterating and task ids are not in series like 1,2,3 more difference like 3,6 and so on  
                                difference_task_id =  active_task_id- task_id_recent-value_iterate
                                value_iterate+=1
                                

                            if first_request:
                                #When sae object bt llm.generate is called again remeber the last id for indexing in create data
                                task_id_recent = active_task_id
                                first_request=False
                                second_request=True
                            # else:

                            #     task_id_recent=active_task_id-task_id_recent
                            if active_task_id in self.skip_task_ids:
                                logging.info(
                                    f"[Cache Used] Skip request  {str(active_task_id)}"
                                )                                
                                continue
                            next_request = APIRequest(
                                task_id=active_task_id,
                                recent_task_id = task_id_recent,
                                difference_task_id= difference_task_id,
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
                                    association_filepath,
                                    project_name,
                                    status_tracker,
                                )
                            )
                        else:
                            # update counters
                            available_request_capacity -= 1
                            available_token_capacity -= next_request_tokens
                            # next_request.attempts_left -= 1

                            asyncio.create_task(
                                next_request.call_api(
                                    session=session,
                                    request_url=request_url,
                                    request_header=request_header,
                                    retry_queue=queue_of_requests_to_retry,
                                    cache_filepath=cache_filepath,
                                    association_filepath= association_filepath,
                                    project_name=project_name,
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
            word in self.url
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
        association_filepath:str,
        project_name:str,
        status_tracker: StatusTracker,
    ):
        async with semaphore:
            await request.call_api(
                session=session,
                request_url=request_url,
                request_header=request_header,
                retry_queue=retry_queue,
                cache_filepath=cache_filepath,
                association_filepath=association_filepath,
                project_name=project_name,
                status_tracker=status_tracker,
                asyncllm_instance=self,
            )

    #create associaton file for first time or if file is deleted
    def create_association_file(self,first=False):
        content={}
        #create association file path
        if os.path.exists(self.cache_dir):
            associatation_filepath = self.cache_dir+"/association.jsonl"
        #first time association file is created with project and requests.jsonl file associations
        if first:
            content['project_requests']={self.project_name: ["request_1"]}
            with open(associatation_filepath, "a") as f:
                json.dump(content,f)
                f.write("\n")
            return
        #Get the association
        dict_pro={}
        if os.path.exists(associatation_filepath):
            with open(associatation_filepath) as f:
                json_data = json.load(f)
                if "project_requests" in list(json_data.keys()):
                        dict_pro["project_requests"] = json_data["project_requests"]
            os.remove(associatation_filepath)
        
        if os.path.exists(self.cache_dir):
            files=os.listdir(self.cache_dir)
            for file in files:
                with open(os.path.join(self.cache_dir, file),"r") as f:
                    data=f.readlines()
                hash_array = []
                for row in data:
                    row=json.loads(row)
                    if row:
                        hash_array.append(list(row[0].keys())[0])
                if len(hash_array)!=0:
                            content[file.split(".")[0]] = hash_array
                else:
                        print("No data found")
        else:
            raise ValueError("Cache file path not proper")
        
        if len(content)!=0:
            request_name_list  = list(content.keys())
            
            if len(dict_pro)!=0:
                content["project_requests"]=dict_pro["project_requests"]
            else:
                #If association file deleted and no informartion abut project all requests assinged to default dataformer project
                content["project_requests"]={"dataformer":request_name_list}
            
            with open(associatation_filepath, "a") as f:
                json.dump(content,f)
                f.write("\n")
        else:
            raise FileNotFoundError("All request cache files empty")
   
            
    

    def generate(
        self,
        request_list: typing.List,
        cache_vars: typing.Dict = {},
        task_id_generator=None,
        use_cache=True,
        project_name=None,
        clear_project_cache=typing.Union[str, typing.List[str]] ,
    ):
        if project_name:
            project_name = project_name.lower()
        if isinstance(clear_project_cache,str):
            if clear_project_cache=="full":
                delete_cache(project_or_full="full",dir_path=self.cache_dir)
            else:
                delete_cache(project_or_full=clear_project_cache.lower(),dir_path=self.cache_dir)
        
        elif isinstance(clear_project_cache,list):
            for project in clear_project_cache:
                #Delete the files of reqestive projects in case of list of projects
                delete_cache(project_or_full=project.lower(),dir_path=self.cache_dir)  

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

        #override the project name
        if project_name is not None:
            self.project_name = project_name
                
        # Check if 'model' is present in all request items
        if not all("model" in request for request in request_list):
            # Since all request already have model, we can ignore self.model for cache.
            # ignore_keys.append("model")
        # else:
            for request in request_list:
                #over ride model based on the api provider or base url in the request list
                if "model" not in request:
                    model=self.model
                    api_provider=""
                    url=""
                    if "url" in list(request.keys()) or "api_provider" in list(request.keys()):
                        if "api_provider" in list(request.keys()):
                            api_provider=request['api_provider']
                        if "url" in list(request.keys()):
                            url=request['url']
                        url = self.get_requesturl_apikey_list(url,api_provider)[0]
                        
                        model = model_dict.get(url)
                    
                    request["model"] = model
                    


        request_url=""
        api_key=""
        different_apis=False

        #check if any provider or base url present in request list
        if any("api_provider" in request or "url" in request for request in request_list):
                different_apis = True
        
        #If no urls or api providers in request means use default in init
        if not different_apis:
            request_url = self.get_request_url()
            for requests in request_list:
                requests['api_provider'] = self.api_provider
                requests['url'] = request_url
                api_key=""
                if "api_key" in list(requests.keys()):
                    api_key=requests['api_key']
                else:
                    api_key = self.get_api_key()
                requests['api_key'] = api_key
        else:
            if all("api_provider" in request or "url" in request for request in request_list):
                # Since all request already have either api_provider or url, we can ignore self.provider or if url (if set) for cache.
                # all requests have either one of api_provider or _url then no need for initatied or default values for cache hash calculation
                # #because they actuallhy wont be used in requests
                
                    ignore_keys.append("api_provider")
                    ignore_keys.append("url")
               
            else:
                
                use_default_or_request_api=[]
                for request in request_list:
                    key_list = list(request.keys())
                    if "api_provider" in key_list or "url" in key_list:
                       
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
                            url = self.get_request_url()
                        else:
                            url,api_key = self.get_requesturl_apikey()
                        
                        api_provider = self.api_provider
                        request_list[i]['api_provider'] = api_provider
                        request_list[i]['url'] = url
                        request_list[i]['api_key'] = api_key
            
            #to skip records, already processed
            skip=True
            for i in range(len(request_list)):
                    api_provider=""
                    url=""
                    api_key=""
                    keys_list=list(request_list[i].keys())
                    #check if any field is missing
                    if "api_provider" not in keys_list or "api_key" not in keys_list or "rl" not in keys_list:
                        skip=False
                        if "api_provider" in keys_list:
                            api_provider=request_list[i]['api_provider']
                        if "url" in keys_list:
                            url=request_list[i]['url']
                        #get the base url
                        url, api_provider = self.get_requesturl_apikey_list(url,api_provider)
                        
                        #if both not found in request list, may be in initialized init
                        if url=="" and api_provider=="":
                            url,api_key = self.get_requesturl_apikey()
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
                            for url_var, env_var in api_key_dict.items():
                                if url_var in url:
                                    api_key  = os.getenv(env_var)
                                    break
                            

                        
                        if api_key=="" or api_key==None:
                            raise ValueError("Api key not provided or INVALID. Provide appropriate API key with api provider or baseurl.")
                    else:
                        skip=True
                        
                    if not skip:
                        if api_provider=="" or url=="":
                            raise ValueError("No api provider or base url provided")
                        else:
                            request_list[i]['api_provider'] = api_provider
                            request_list[i]['url'] =url
                            request_list[i]['api_key'] = api_key
                   

        for request in request_list:
            for key, value in self.sampling_params.items():
                if key not in request:
                    request[key] = value

        if task_id_generator:
            self.task_id_generator = task_id_generator
        else:
            #Incase if llm.enerate called multiple times on same object
            self.task_id_generator = None
        self.response_list = []
        # check if requests_list have properly assigned model parameter 
        if different_apis:
            for request in request_list:
                self.check_model_exists(request['url'],request['api_provider'],request['api_key'],request['model'])
        # If cache, add cached responses to self.response_list
        # if "request_details" not in cache_vars:
            # self.request_details = get_request_details(request_list)
       
        
        # create cache list
        new_re_cache = [create_hash(get_cache_vars(re.copy(), ignore_keys=ignore_keys, additional_data=cache_vars,vars_use=False)) for re in request_list]
        if self.gen_type=="text":
                    for i in request_list:
                        if i["api_provider"]=="groq":
                            raise ValueError("Groq doesn't support text generation.")
        # We create cache_hash to sort the async responses even when use_cache is False.
        self.cache_hash = new_re_cache

        self.skip_task_ids = []
       
        cache_filepath=None
        association_filepath=None
        if use_cache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            
            #check if files exists, to assign names as first request or next request in order
            files = os.listdir(self.cache_dir)
            files.sort(key=lambda x: x.split('.')[0])
            #If someone deletes all the files except the association file, then delete it 
            if len(files)==1 and "association.jsonl" in files[0]:
                os.remove(os.path.join(self.cache_dir,files[0]))
                files=[]
            if len(files)==0:
                filename="request_1"
            else:                
                filename= "request_"+ str(int(files[-1].split(".")[0].split("_")[1])+1)
        
            #name the cache and association file
            cache_filepath = f"{self.cache_dir}/{filename}.jsonl"
            association_filepath =f"{self.cache_dir}/association.jsonl"
            
            #create if the association file doesn't exist or is deleted or corrupted
            if filename!="request_1":
                #create association file for first time
                if not os.path.exists(association_filepath):
                    self.create_association_file()

                else:
                    #corrupted or edited file inaccurately case
                    with open(association_filepath,"r") as f:
                        data_json = json.load(f)
                    keys_files= list(data_json.keys())
                    files = os.listdir(self.cache_dir)
                    #Check if all requests hash ids present in association if not possibility of edited file, create new 
                    if not all(filename in files or "project_requests" for filename in keys_files):
                        self.create_association_file()
            
                
                if os.path.exists(association_filepath):
                    #find the overlapping caches and skip them
                    cached_indices = []
                    with open(association_filepath,"r") as file:
                        cache_hash_data = json.load(file)
                   
                    values=[]
                    new_cache_hash_list = self.cache_hash.copy()
                    
                    for key in cache_hash_data:
                        #skip this keywors as it represent project association
                        if key=="project_requests":
                            continue
                        #get the stored hash from file
                        stored_hash_list = cache_hash_data[key]
                        
                        indices=[]
                        #iterate and check hashes
                        for i, hashh in enumerate(new_cache_hash_list):
                            if i not in cached_indices:
                                if hashh in stored_hash_list:
                                
                                    values.append(hashh)                             
                                    cached_indices.append(i)                                                                
                                    indices.append(stored_hash_list.index(hashh))
                        #read data and save response in response_list 
                        with open(os.path.join(self.cache_dir,key+".jsonl")) as f:
                            file_data = f.readlines()
                            #Get the responses
                            for j in indices:
                                self.response_list.append(json.loads(file_data[j]))  
                                      
                    #cached indices to be skipped
                    self.skip_task_ids = cached_indices
                else:
                    print("Association file doesn't exist")
            else:
                self.create_association_file(first=True)
        
        
        if different_apis:
            #for different api providers
            asyncio.run(
                self.process_api_requests(
                    request_url,
                    api_key="",
                    request_list=request_list,
                    cache_filepath=cache_filepath,
                    association_filepath=association_filepath,
                    project_name=self.project_name,
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
                    association_filepath=association_filepath,
                    project_name=self.project_name,
                    different_requests=False
                )
            )           
        sorted_response_list = sorted(
            self.response_list, key=lambda x: list(x[0].values())[0]
        )
        
        sorted_response_list = [
            item[1:] for item in sorted_response_list
        ]  # Exclude self.cache_hash from the list
        
        return sorted_response_list
