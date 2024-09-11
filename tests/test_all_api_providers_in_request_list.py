#tests\test_all_api_providers_in_request_list.py
#The test is created for two purposes
#One: Test all api_providers, Two: Test smooth working when all requests contain api provider

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM


def test_all_api_providers():
    # Assuming llm.generate returns a response

    api_providers=["openai", "groq", "monsterapi", "together", "deepinfra", "openrouter","anthropic"]
    many_providers_text=[]
    many_providers_chat=[]
    for i in api_providers:
        request_message = {"messages": [{"role": "user", "content": "Give only names of 3 places to visit in India."}]}
        request_text = {"prompt": "Complete the paragraph.\n She lived in nashville"}
        
        request_message['api_provider'] = i
        if i!="groq":
            request_text['api_provider'] = i
        if i!="anthropic":
            many_providers_text.append(request_text)
        many_providers_chat.append(request_message)

    llm=AsyncLLM(gen_type="text")
    response = llm.generate(many_providers_text, project_name="NewProject")
    # Add assertions to check the proper response
    assert response is not None,"Whole Response is None"
    assert len(response) == len(many_providers_text),"Response and instruction length different"
    assert response[0][1]["choices"][0]["text"] is not None,"Response not containing text result"
   
    llm=AsyncLLM(gen_type="chat")
    response = llm.generate(many_providers_chat, project_name="NewProject")
    # Add assertions to check the proper response
    assert response is not None,"Response is None"
    assert len(response) == len(many_providers_chat),"Response and instruction length different"
    assert response[0][1]["choices"][0]["message"] is not None,"Response not containing text result"
