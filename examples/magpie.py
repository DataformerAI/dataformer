import os
import json
from dataformer.components import MAGPIE
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("JARVIS_API_KEY")

# Ollama - openai compatible endpoint
# Deploy via https://jarvislabs.ai/serverless/frameworks/ollama_serverless
URL = "https://api.jarvislabs.net/openai/900f66f0cd836278/v1/chat/completions"

question_prompt = """
"You are an AI assistant specialized in logical thinking and problem-solving.
Your purpose is to help users work through complex ideas, analyze situations, and draw conclusions based on given information.
Approach each query with structured thinking, break down problems into manageable parts, and guide users through the reasoning process step-by-step."
"""

answer_prompt = """
You're an AI assistant that responds to the user with maximum accuracy.
To do so, your first task is to think what the user is asking for, thinking step by step.
During this thinking phase, you will have reflections that will help you clarifying ambiguities.
In each reflection you will list the possibilities and finally choose one. Between reflections, you can think again.
At the end of the thinking, you must draw a conclusion.
You only need to generate the minimum text that will help you generating a better output, don't be verbose while thinking.
Finally, you will generate an output based on the previous thinking.

This is the output format you have to follow:

```
<thinking>
Here you will think about what the user asked for.
</thinking>

<reflection>
This is a reflection.
</reflection>

<reflection>
This is another reflection.
</reflection>

</thinking>

<output>
Here you will include the output.
</output>
```
""".lstrip()

sampling_params = {"temperature": 0.6, "top_p": 1}
llm = AsyncLLM(model="llama3.1", url=URL, sampling_params=sampling_params, max_requests_per_minute=5, api_key=api_key)

# Only user template, NOT the entire chat template.
template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
magpie = MAGPIE(llm, template=template, question_prompt=None, answer_prompt=None)
num_samples=5 # Actual samples can be bit less since we filter out any empty responses.
dataset = magpie.generate(num_samples)

with open("dataset.json", "w") as json_file:
    json.dump(dataset, json_file, indent=4, ensure_ascii=False)