from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

llm = AsyncLLM(url="https://api.deepinfra.com/v1/openai/embeddings", model="thenlper/gte-large")

instruction = "hey" 
data_dict = {
        "input": instruction,
        # "encoding_format": "float"
}

request_list = [data_dict]
response_list = llm.generate(request_list)

print(response_list)