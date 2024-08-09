import json

from dataformer.components import MAGPIE
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Ollama - openai compatible endpoint
# Template - https://jarvislabs.ai/templates/ollama
# Once you create an instance, click on API - to get Endpoint URL & Deploy as such: https://jarvislabs.ai/blogs/ollama_deploy (ollama pull llama3)
URL = "https://a8da29c1850e1.notebooksa.jarvislabs.net/v1/chat/completions"

sampling_params = {"temperature": 0.6, "top_p": 1}
llm = AsyncLLM(model="llama3", url=URL, sampling_params=sampling_params, api_provider="ollama", max_requests_per_minute=5)

# Only user template, NOT the entire chat template.
template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
magpie = MAGPIE(llm, template=template)
num_samples=5 # Actual samples can be bit less since we filter out any empty responses.
dataset = magpie.generate(num_samples)

with open("dataset.json", "w") as json_file:
    json.dump(dataset, json_file, indent=4)
