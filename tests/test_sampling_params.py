#tests\test_sampling_params.py
#This test is created to check if all the sampling paramters work perfectly
from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list

from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
def test_sampling_parameters():
    dataset = load_dataset("dataformer/self-knowledge")
    datasetsub = dataset["train"].select(range(5))
    instructions = [example["question"] for example in datasetsub]


    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    request_list = get_request_list(instructions, sampling_params)

    llm = AsyncLLM(api_provider="groq")
    response_list = llm.generate(request_list)
    assert response_list is not None,"Resonse Empty"
    assert all(response[1]["choices"][0]['message']['content'] for response in response_list), "Response not generated poperly"
    