# tests\test_ollama.py
#This Test is to check the smooth workng of olloma
# from dataformer.llms import AsyncLLM
# from dotenv import load_dotenv
# Load environment variables from .env file
# load_dotenv()
# # Ollama - openai compatible endpoint
# # Template - https://jarvislabs.ai/templates/ollama
# # Once you create an instance, click on API - to get Endpoint URL & Deploy as such: https://jarvislabs.ai/blogs/ollama_deploy (ollama pull llama3)
# def test_ollama():
#     URL = "https://a8da29c1850e1.notebooksa.jarvislabs.net/v1/chat/completions"

#     sampling_params = {"temperature": 0.6, "top_p": 1}
#     llm = AsyncLLM(model="llama3", url=URL, sampling_params=sampling_params, api_provider="ollama", max_requests_per_minute=5)

#     request_list = [{"messages": [{"role": "user", "content": "Hi there!"}], "stream": False},
#                     {"messages": [{"role": "user", "content": "Who are you?"}], "stream": False}]

#     response_list = llm.generate(request_list)

#     assert response_list is not None,"Response is None"
#     assert all("content" in list(i[1]["choices"][0]["message"].keys()) for i in response_list),"Response returned doesn't contain generation answers" 

