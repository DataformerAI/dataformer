# tests\test_various_limits_request_list_formation.py
# The test is created to ensure various limits on requests are applied and if the get_request_details function creates appropriate requests
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list
from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

def test_various_limits_and_request_lists():

    dataset = load_dataset("dataformer/self-knowledge")
    datasetsub = dataset["train"].select(range(1))
    instructions = [example["question"] for example in datasetsub]


    request_list = get_request_list(instructions)
    assert request_list is not None,"response list is None"
    assert all(isinstance(i,dict) for i in request_list),"Request_list not formateed properly, individual elements not dictionary"
    assert all("messages" in list(i.keys()) for i in request_list),"Request_list not pformatted properly, message key is missing"
    assert all(isinstance(i['messages'],list) for i in request_list),"Request_list not formatted properly, messages value must be a list"
    assert all(isinstance(j,dict) for i in request_list for j in i['messages']),"Request_list not formatted properly, messages list must have conversaton dicts"
    assert all("role" in list(j.keys()) and "content" in list(j.keys()) for i in request_list for j in i['messages']),"Request_list not formatted properly, messages list must have conversaton dicts wih role and content keys"
# request_list = [
    #     {"messages": [{"role": "user", "content": prompt}], "temperature": 0.7} for prompt in instructions
    # ]


    llm = AsyncLLM(api_provider="deepinfra",max_requests_per_minute=200,max_concurrent_requests=200,max_attempts=2,max_tokens_per_minute=100)
    response_list = llm.generate(request_list)
    assert response_list is not None,"response list is None"
    assert all(response[1]["choices"][0]["message"]["content"] for response in response_list),"Response not returned to chat"

    llm = AsyncLLM(api_provider="together")
    response_list = llm.generate(request_list)
    assert response_list is not None,"response list is None"
    assert all(response[1]["choices"][0]["message"]["content"] for response in response_list),"Response not returned to chat"