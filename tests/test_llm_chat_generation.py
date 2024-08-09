#tests\test_llm_chat_generation.py
#This test is created to check if llm generation for chats works perfectly.
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM

def test_llm_generate_proper_response():
    request_list = [
        {"messages": [{"role": "user", "content": "Why should people read books?"}]},
        {"messages": [{"role": "user", "content": "Name people who have won medals at Olympics 2024."}]}
    ]
    llm=AsyncLLM(api_provider="together")
    # Assuming llm.generate returns a response
    response = llm.generate(request_list, project_name="NewProject")
    # Add assertions to check the proper response
    assert response is not None,"response is None"
    assert "content" in list(response[0][1]["choices"][0]["message"].keys()),"No Response of chat"
    assert "content" in list(response[1][1]["choices"][0]["message"].keys()),"No Response of chat"