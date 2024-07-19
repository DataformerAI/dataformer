import json

from dataformer.components.magpie import MAGPIE
from dataformer.llms import AsyncLLM

# For now, this works only with ollama and llama3 / Phi
URL = "https://kal97043ykgygf-11434.proxy.runpod.net/api/chat"
# URL = "https://ruchimali--fastapi-ollama-f.modal.run/api/chat"

llm = AsyncLLM(
    model="llama3",
    base_url=URL,
    api_provider="ollama",
)
magpie = MAGPIE(llm)

### Default Parameters
# { "seed": 676,
#   "temperature": 0.8,
#   "top_p": 1 }

# Define custom parameters
params = {"seed": 700, "temperature": 0.6, "top_p": 1}

# This automatically cleans empty responses
dataset = magpie.generate(5, params)

with open("dataset.json", "w") as json_file:
    json.dump(dataset, json_file, indent=4)
