from dataformer.llms import AsyncLLM

llm = AsyncLLM(url="https://api.deepinfra.com/v1/openai/embeddings", model="thenlper/gte-large") # api_key="<your-api-key-here>"

instruction = "hey" 
data_dict = {
        "input": instruction,
        # "encoding_format": "float"
}

request_list = [data_dict]
response_list = llm.generate(request_list)

print(response_list)