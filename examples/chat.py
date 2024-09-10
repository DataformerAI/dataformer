from dataformer.llms import AsyncLLM
from dataformer.utils import get_request_list

from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

COLOR = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "ENDC": "\033[0m",
}

dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(5))
instructions = [example["question"] for example in datasetsub]


sampling_params = {"temperature": 0.7}
request_list = get_request_list(instructions, sampling_params)

# request_list = [
#     {"messages": [{"role": "user", "content": prompt}], "temperature": 0.7} for prompt in instructions
# ]

"""
API Providers
- openai
- groq
- together
- deepinfra
- openrouter
"""

llm = AsyncLLM(api_provider="groq")
response_list = llm.generate(request_list)

for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"{COLOR['BLUE']}Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer: {answer}{COLOR['ENDC']}\n")

