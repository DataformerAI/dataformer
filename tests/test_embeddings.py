#tests\test_embeddings.py
#This Test is created to test if the embeddings are generated properly given the url.
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

def test_embedding_generation():
    instruction = "hey"
    data_dict = {
        "input": instruction,
        # "encoding_format": "float"
    }
    llm = AsyncLLM(url="https://api.deepinfra.com/v1/openai/embeddings", model="thenlper/gte-large")
    request_list = [data_dict]
    response_list = llm.generate(request_list)
    assert response_list is not None,"Response is None"
    #check if embeddings exists
    assert "embedding" in list(response_list[0][1]['data'][0].keys()),"Embddings not found in the Response"

# Run the test by executing `pytest embedding_test.py` in the terminal