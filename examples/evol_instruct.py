from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

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

print("\n\n")
for item in results:
    print(f"{COLOR['BLUE']}Original Instruction: {item['original_instruction']}{COLOR['ENDC']}")
    for evolved_instruction in item['evolved_instructions']:
        print(f"{COLOR['GREEN']}Evolved Instruction: {evolved_instruction}{COLOR['ENDC']}")
    for answer in item['answers']:
        print(f"{COLOR['PURPLE']}Answer: {answer}{COLOR['ENDC']}")
    print("\n")
