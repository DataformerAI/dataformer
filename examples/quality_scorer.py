from dataformer.components import QualityScorer
from dataformer.llms import AsyncLLM
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

input = [{"instruction": "What are the key features of Python programming language?",
          "responses": ["Python is known for its simplicity, readability, and versatility. It supports multiple programming paradigms, has a rich standard library, and is widely used in various domains such as web development, data science, and automation.",
                        "Python is a language. It is used for coding. Some people like it. It can do things. There are libraries. It is not the only language. Some say it's good, others not so much.",
                        "Python is a popular programming language with easy syntax and extensive libraries. It is used for tasks like scripting, web development, and scientific computing. Its dynamic typing can be both a strength and a weakness depending on the context."
                        ]}]

llm = AsyncLLM(
    model="gpt-4o", api_provider="openai"
)

scorer = QualityScorer(
    llm=llm
)

results = scorer.score(
    input, use_cache=False
    ) # By default cache is True.

for result in results:
    instruction = result['instruction']
    responses = result['responses']
    scores = result['scores']
    print(f"{COLOR['BLUE']}Instruction: {instruction}{COLOR['ENDC']}")
    for i in range(len(responses)):
        print(f"{COLOR['PURPLE']}Response{i+1}: {responses[i]}{COLOR['ENDC']}")
        print(f"{COLOR['GREEN']}Score{i+1}: {scores[i]}{COLOR['ENDC']}")
    print("\n")