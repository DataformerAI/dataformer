from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

#Mixture of Agents technique is used achieve amazing performance and results with 
#It employs a layered architecture where each layer comprises of multiple llms.

#In this example a two layered approach is employed where layer 1 comprises of different llms
#generating answers to a user query and these answers from multiple models are then supplied to
#aggregator model (llm) of layer 2 which critically evaluates, corrects any biases and synthesis the responses into one single, more accurate response

#define the keys
deepinfra_api_key=""
openai_api_key=""

#Define the reference models, their providers and keys, these llms act as models or llms of layer 1
reference_models_providers = {
    "mistralai/Mixtral-8x22B-Instruct-v0.1":["deepinfra",deepinfra_api_key],
    "gpt-4o":["openai",openai_api_key]
    # Add more reference models here if needed
}

 #Colors for printing the output
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

#Define the aggregator model and the system prompt required for giving the final response 
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models
to the latest user query. Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it
may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined,
accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and
adheres to the highest standards of accuracy and reliability.

Responses from models: """

#Specify the api provider for aggregator model
api_provider="deepinfra"

#Define the aggregator llm
aggregator_llm = AsyncLLM(api_provider=api_provider, model=aggregator_model)

#Defining the user prompts, can define any required prompts
request_list = [
                {
                    "messages":
                    [
                        {
                            "role": "user", 
                        "content": "Give only names of 3 places to visit in India."
                        }
                    ]
                },
                {
                    "messages":
                    [
                        {
                            "role": "user", 
                            "content": "Give only names of any 3 fiction books."
                        }
                    ]
                }
                ]


#Creating AsyncLLM object to provide different model responses to the user query
llm = AsyncLLM()

#Adding the models defined above
#creating new requests list which will have the user query with the required models mentioned
final_request_list=[]
for models in reference_models_providers:
    for request in request_list:
        new =request.copy()
        new["model"]= models #Adding the respective model
        new["api_provider"]=reference_models_providers[models][0]
        new["api_key"] = reference_models_providers[models][1]
        final_request_list.append(new)
      

#Collect responses from the reference llms, here requests have different models, hence, responses from different model for a given user query are collected.
reference_models_response_list = llm.generate(final_request_list)

#store the processed responses from different models, for passing to the aggregator LLM in prompt later
reference_models_results=[]

print(f"{COLOR['RED']}Models Individual Responses{COLOR['ENDC']}")

#Model incrementer for iterating each model
model_incr=0

#Create reference_models_results for storing all models responses  to a spcific user query
for i in range(len(reference_models_providers)):
    reference_models_results.append([])

#iterating on 2 increments as 2 models in list
for i in range(0,len(reference_models_response_list),len(reference_models_providers)):
    #the answers incrementer for answers given to queries by each model
    answer_incr=0
    response_list = reference_models_response_list[i:i+len(reference_models_providers)]
    
    for request, response in zip(request_list, response_list):
        prompt = request["messages"][0]["content"]
        answer = response[1]["choices"][0]["message"]["content"]
        print(f"{COLOR['BLUE']}Reference Model: {model_incr}\n Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")
        
        #In each list have model's responses to a query.
        reference_models_results[answer_incr].append(str(model_incr)+"... "+answer) #append
        answer_incr+=1
    model_incr+=1

#pass the responses of models to the aggregator llm in the prompt with the respective user query 
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
for request, response in zip(request_list, response_list_aggregator):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"{COLOR['BLUE']}Prompt: {prompt}{COLOR['ENDC']}\n{COLOR['GREEN']}Answer:\n {answer}{COLOR['ENDC']}\n")