# tests/test_llm_generate.py
#Test is created to check if additional cache vars are taken into consideration
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM

def test_cache_vars():
    request_list = [{"prompt": "Complete the paragraph.\n She lived in nashville"},{"prompt": "Write a story on 'Homesty is the best Policy'"}]

    llm=AsyncLLM(api_provider="together",gen_type="text")
    # Assuming llm.generate returns a response
    response_list = llm.generate(request_list, project_name="NewProject",cache_vars={"project_id":32})
    assert response_list is not None,"Response nis Noen"
    assert all(response[1]["choices"][0]['text'] for response in response_list),"Some of the Response not containing text result"
