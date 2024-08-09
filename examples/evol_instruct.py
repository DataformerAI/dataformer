from dataformer.components.evol_instruct import EvolInstruct
from dataformer.llms import AsyncLLM
from datasets import load_dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
COLOR = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "ENDC": "\033[0m",
}

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
    for evolved_instruction, answer in zip(item['evolved_instructions'], item['answers']):
        print(f"{COLOR['GREEN']}Evolved Instruction: {evolved_instruction}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Answer: {answer}{COLOR['ENDC']}")
    print("\n")