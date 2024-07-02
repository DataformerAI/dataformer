from dataformer.llms.openllm import OpenLLM
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(50))
instructions = [example["question"] for example in datasetsub]


request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
]

#Groq Provider
#llm = OpenLLM(model = "mixtral-8x7b-32768",api_provider="groq"
#----------------------------------------------------------------------------------------------------------------------------------------
#Anthropic
#llm = OpenLLM(model = "claude-2.1",api_provider="anthropic")
#---------------------------------------------------------------------------------------------------------------------------------------
#together
#llm = OpenLLM(model = "mistralai/Mixtral-8x7B-Instruct-v0.1",api_provider="together")
#---------------------------------------------------------------------------------------------------------------------------------------
#Anyscale
#llm = OpenLLM(model = "mistralai/Mistral-7B-Instruct-v0.1",api_provider="anyscale")
#---------------------------------------------------------------------------------------------------------------------------------------
#DeepInfra
#llm = OpenLLM(model = "meta-llama/Meta-Llama-3-8B-Instruct",api_provider="deepinfra")
#---------------------------------------------------------------------------------------------------------------------------------------
#OpenRouter
#llm = OpenLLM(model = "openai/gpt-3.5-turbo",api_provider="openrouter")
#---------------------------------------------------------------------------------------------------------------------------------------
#Ollama
#llm = OpenLLM(model="llama3", api_provider="ollama")
#---------------------------------------------------------------------------------------------------------------------------------------
#OpenApi
llm = OpenLLM(model="gpt-3.5-turbo")


response_list = llm.generate(request_list)

for request, response in zip(request_list, response_list):
    prompt = request['messages'][0]['content']
    answer = response[-1]["choices"][0]["message"]["content"]
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")
