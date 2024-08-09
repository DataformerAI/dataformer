# tests\test_evol_instruct.py
#The test is created to check if the eval_instruct component works perfectly.
from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
def test_evol_instruct():
    dataset = load_dataset("dataformer/self-knowledge")
    datasetsub = dataset["train"].select(range(2))
    instructions = [example["question"] for example in datasetsub]

    llm = AsyncLLM(
        model="mixtral-8x7b-32768", api_provider="groq"
    )  # Make sure you have set "GROQ_API_KEY" in .env file.

    evol_instruct = EvolInstruct(
        llm=llm,
        num_evolutions=2,  # Number of times to evolve each instruction
        store_evolutions=True,  # Store all evolutions
        generate_answers=True,  # Generate answers for the evolved instructions
        # include_original_instruction=True  # Include the original instruction in the results
    )
    results = evol_instruct.generate(
        instructions, use_cache=False
    )  # By default cache is True.

    assert results is not None,"Response is None"
    for item in results:
        assert item is not None,"A response is missing"
        assert all(i in list(item.keys()) for i in ["original_instruction","evolved_instructions","answers"]),"Some keys missing in response"
        assert item['original_instruction'] is not None ,"original_instruction value not found in response"
        assert item['evolved_instructions'] is not None ,"evolved_instructions value not found in response"
        assert item["answers"] is not None,"answers value not found in response"