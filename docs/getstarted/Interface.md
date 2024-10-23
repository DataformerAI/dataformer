# Interface Documentation

## Overview
This documentation provides an overview of how to use the `cot`, `SelfConsistency`, `pvg`, and `rto` classes from the `dataformer` library to generate responses based on user queries. Each class utilizes a language model (LLM) to provide answers to specific types of questions.

## Initialization
Before using the classes, ensure that you have loaded the necessary environment variables and initialized the language model.

### Example Initialization
```python
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = AsyncLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    api_provider="deepinfra"
)
```

## Using the `cot` Class
The `cot` class implements a Chain of Thought (CoT) approach for generating responses.

### Example Input
```python
request_list = [
    {
        "messages": [
            {
                "role": "user", 
                "content": "If a train leaves a station traveling at 60 miles per hour and another train leaves the same station 30 minutes later traveling at 90 miles per hour, when will the second train catch up to the first train?"
            }
        ]
    }
]

cot_instance = cot(llm=llm)
results = cot_instance.generate(request_list)

# Print results
for item in results:
    print(f"Cot Answer: {item['cot_response']}")
    print(f"Model Answer: {item['model_response']}")
```

### Example Output
```
Prompt: If a train leaves a station traveling at 60 miles per hour and another train leaves the same station 30 minutes later traveling at 90 miles per hour, when will the second train catch up to the first train?

Cot Answer: The second train will catch up to the first train after 1 hour and 30 minutes.
Model Answer: The second train will catch up to the first train after 1.5 hours.
```

## Using the `SelfConsistency` Class
The `SelfConsistency` class generates answers based on self-consistency checks.

### Example Input
```python
request_list = [
    {
        "messages": [
            {
                "role": "user", 
                "content": "I have a dish of potatoes. The following statements are true: No potatoes of mine, that are new, have been boiled. All my potatoes in this dish are fit to eat. No unboiled potatoes of mine are fit to eat. Are there any new potatoes in this dish?"
            }
        ]
    }
]

self_consistency_instance = SelfConsistency(llm=llm)
results = self_consistency_instance.generate(request_list=request_list, return_model_answer=True)

# Print results
for item in results:
    print(f"Self Consistency Answer: {item['SelfConsistency_response']}")
    print(f"Model Answer: {item['model_response']}")
```

### Example Output
```
Prompt: I have a dish of potatoes. The following statements are true: No potatoes of mine, that are new, have been boiled. All my potatoes in this dish are fit to eat. No unboiled potatoes of mine are fit to eat. Are there any new potatoes in this dish?

Self Consistency Answer: No, there are no new potatoes in the dish.
Model Answer: There are no new potatoes in the dish based on the given statements.
```

## Using the `pvg` Class
The `pvg` class generates and verifies solutions to problems.

### Example Input
```python
request_list = [
    {
        "messages": [
            {
                "role": "user", 
                "content": "Write a code in python for timetable generation. Consider all the constraints."
            }
        ]
    }
]

pvg_instance = pvg(llm=llm)
results = pvg_instance.generate(request_list)

# Print results
for item in results:
    print(f"PVG Answer: {item['pvg_response']}")
    print(f"Model Answer: {item['model_response']}")
```

### Example Output
```
Prompt: Write a code in python for timetable generation. Consider all the constraints.

PVG Answer: 
```python
def generate_timetable(classes):
    # Implementation for generating a timetable
    pass
```
Model Answer: The code provided is a skeleton for generating a timetable based on given classes.
```

## Using the `rto` Class
The `rto` class implements a round trip optimization approach for generating code.

### Example Input
```python
request_list = [
    {
        "messages": [
            {
                "role": "user", 
                "content": "Write a genetic algorithm code in python which is fast."
            }
        ]
    }
]

rto_instance = rto(llm=llm)
results = rto_instance.generate(request_list)

# Print results
for item in results:
    print(f"RTO Answer: {item['rto_response']}")
    print(f"Model Answer: {item['model_response']}")
```

### Example Output
```
Prompt: Write a genetic algorithm code in python which is fast.

RTO Answer: 
```python
def genetic_algorithm():
    # Implementation of a fast genetic algorithm
    pass
```
Model Answer: The provided code outlines a genetic algorithm structure that can be optimized for speed.
```

## Conclusion
This documentation provides a comprehensive guide on how to use the `cot`, `SelfConsistency`, `pvg`, and `rto` classes to generate and verify responses based on user queries. Each class serves a specific purpose and can be utilized to enhance the interaction with the language model.
