from dataformer.components import QualityScorer
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv

load_dotenv()

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

print("\n\n")
for result in results:
    instructions = result['instructions']
    scores = result['scores']
    for i in range(max(len(instructions), len(scores))):
        if i < len(instructions):
            print(f"{COLOR['BLUE']}Instruction: {instructions[i]}{COLOR['ENDC']}")
        if i < len(scores):
            print(f"{COLOR['GREEN']}Score: {scores[i]}{COLOR['ENDC']}")
    print("\n")
