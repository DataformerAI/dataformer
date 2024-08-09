#tests\test_eval_quality.py
#The test is created to chekc if the eval_quality compnent works perfectly.
from dataformer.components.evol_quality import EvolQuality
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

def test_evol_quality():
    dataset = load_dataset("dataformer/self-knowledge")
    datasetsub = dataset["train"].select(range(2))
    instructions = [example["question"] for example in datasetsub]


    llm = AsyncLLM(
        model="mixtral-8x7b-32768", api_provider="groq"
    )  # Make sure you have set "GROQ_API_KEY" in .env file.

    # Generating answers for the question first
    request_list = [
        {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
    ]
    answers = llm.generate(request_list, use_cache=True)
    answers = [answer[1]["choices"][0]["message"]["content"] for answer in answers]

    # Formatting in required format for EvolQuality
    inputs = [
        {"instruction": instruction, "response": response}
        for instruction, response in zip(instructions, answers)
    ]

    evol_quality = EvolQuality(
        llm=llm,
        num_evolutions=1,  # Number of times to evolve each response
        store_evolutions=True,  # Store all evolutions, if False only final evolution is shown.
        include_original_response=False,  # Include the original response in evolved_responses
    )
    results = evol_quality.generate(inputs, use_cache=False)

    assert results is not None,"Response is None"
    for result in results:
        assert result is not None,"A resposne is missing"
        assert all(i in list(result.keys()) for i in ["instruction","response","evolved_responses"]),"Some keys missings in response"
        assert result['instruction'] is not None,"instruction value missing in response returned"
        assert result['response'] is not None,"response value missing in response returned"
        assert result['evolved_responses'] is not None,"evolved_responses value mising in response returned"