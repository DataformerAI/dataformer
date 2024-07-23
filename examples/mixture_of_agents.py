from dataformer.llms import AsyncLLM

#Define the reference models
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct"
    # Add more reference models here if needed
]

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
#Define the aggregator model and the system prompt
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models
to the latest user query. Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it
may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined,
accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and
adheres to the highest standards of accuracy and reliability.

Responses from models: """

#Defining the user prompts
request_list = [
                {
                    "messages":
                    [
                        {
                            "role": "user", 
                        "content": "List any 3 places of cultural importance to visit in India."
                        }
                    ]
                },
                {
                    "messages":
                    [
                        {
                            "role": "user", 
                            "content": "Give any 3 best books to refer to learn about the history of India."
                        }
                    ]
                }
                ]

#store the reference llms initialized objects
reference_models_llms=[]

#Specify the api provider
api_provider="deepinfra"

#create the reference llms
for model in reference_models:
    reference_models_llms.append(AsyncLLM(api_provider=api_provider, model=model))

#Define the aggregator llm
aggregator_llm = AsyncLLM(api_provider=api_provider, model=aggregator_model)

#Collect responses from the reference llms
reference_models_response_list=[]
for llm in reference_models_llms:
    reference_models_response_list.append(llm.generate(request_list))

#store and print, the processed responses for passing to the aggregator LLM 
reference_models_results=[]

print(f"{COLOR['RED']}Models Individual Responses{COLOR['ENDC']}")
model_incr=0
for response_list in reference_models_response_list:
    #the answers incrementer
    answer_incr=0
    
    for request, response in zip(request_list, response_list):
        prompt = request["messages"][0]["content"]
        answer = response[1]["choices"][0]["message"]["content"]
        print(f"{COLOR['BLUE']}Reference Model: {model_incr}\n Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")
        if len(reference_models_results)!=len(request_list):
            reference_models_results.append([str(answer_incr)+"... "+answer])
        else:
            reference_models_results[answer_incr].extend([str(answer_incr)+"... "+answer])
        answer_incr+=1
    model_incr+=1
    
request_list_aggregator=[]
for i in range(len(request_list)):
    request_list_aggregator.append({"messages":
                    [
                        {
                            "role": "system", 
                            "content": aggregator_system_prompt
                            +"\n"
                            +"\n".join(reference_models_results[i])}, #aggregate all the responses to user prompt from all models
                        {
                            "role": "user", 
                            "content": request_list[i]["messages"][0]["content"] # The user prompt
                        }
                ]
    })

#Generate the response from the aggregator llm
response_list_aggregator = aggregator_llm.generate(request_list_aggregator)

#print the response from the aggregator llm
print(f"{COLOR['RED']}Aggregator Model's Response{COLOR['ENDC']}")
for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"{COLOR['BLUE']}Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")

