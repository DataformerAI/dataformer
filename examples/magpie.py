import json

from dataformer.components.magpie import MAGPIE

URL = "https://kal97043ykgygf-11434.proxy.runpod.net/api/chat"
model = "llama3"

magpie = MAGPIE(model, URL)

### Default Parameters
# { "seed": 676,
#   "temperature": 0.8,
#   "top_p": 1 }

# Define custom parameters
params = {"seed": 700, "temperature": 0.6, "top_p": 1}

dataset = magpie.generate(5, params, verbose=True)

with open("dataset.json", "w") as json_file:
    json.dump(dataset, json_file, indent=4)
