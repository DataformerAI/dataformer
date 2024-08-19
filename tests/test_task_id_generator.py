# tests/test_llm_generate.py
#This Test is created to check if task_id_geenrator passed externally works perfectly
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM

#task_id_generator_function for generation of ids
def task_id_generator_function():
    """Generate integers 0, 3, 6, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 3
request_list = [{"prompt": "Complete the paragraph.\n She lived in nashville"},{"prompt": "Write a story on 'Honesty is the best Policy'"},{"prompt": "Write a story on 'There once was a boy'"}]

def test_task_id_generator():
    llm=AsyncLLM(api_provider="together",gen_type="text")
    # Assuming llm.generate returns a response
    response_list = llm.generate(request_list, project_name="NewProject",task_id_generator=task_id_generator_function())
    assert response_list is not None,"Response is None"
    assert all(response[1]["choices"][0]["text"] for response in response_list),"Response not generated properly, text value missing"
