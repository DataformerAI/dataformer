# tests\test_llm_text_generation.py
#This Test is created to check if llm generation on prompts works properly
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM

def test_llm_generate_proper_response():
    request_list = [{"prompt": "Complete the paragraph.\n She lived in nashville"}]
    
    llm=AsyncLLM(api_provider="together",gen_type="text")
    # Assuming llm.generate returns a response
    response = llm.generate(request_list, project_name="NewProject")
    # Add assertions to check the proper response
    assert response is not None,"Response is missing"
    assert response[0][1]["choices"][0]["text"] is not None,"No Response of prompt"
