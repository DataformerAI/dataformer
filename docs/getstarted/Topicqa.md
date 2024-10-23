# Creating Questions and Answers from Topics

## Overview
This documentation provides a guide on how to generate questions and answers based on predefined topics using the `AsyncLLM` class from the `dataformer.llms` module. The process involves creating focused questions that delve into specific aspects of each topic and then generating comprehensive answers for those questions.

## Prerequisites
Before you begin, ensure you have the following:
- Python installed on your machine.
- The `dataformer` library installed. You can install it using pip:
  ```bash
  pip install dataformer
  ```
- The `python-dotenv` library to manage environment variables:
  ```bash
  pip install python-dotenv
  ```

## Setup Instructions

1. **Create a `.env` File**: 
   Create a file named `.env` in your project directory. This file will store your environment variables, such as your OpenAI API key.

2. **Load Environment Variables**: 
   Use the `load_dotenv()` function to load the variables from the `.env` file. This is essential for managing sensitive information.

## Example Code
Below is a sample code snippet to demonstrate how to create questions and answers from a list of topics:

```python
import random
from dataformer.llms.asyncllm import AsyncLLM
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY") 
llm = AsyncLLM(api_key=api_key, api_provider="openai")

# Define a list of topics
topics = [
    "Business and Economics - entrepreneurship, economic theory, investment strategies, marketing, supply chain management",
    "Historical Studies - industrial revolution, global history, military history, ancient civilizations, renaissance",
    "Natural Sciences - botany, chemistry, astronomy, marine biology, physics",
    "Mathematics - number theory, statistics, calculus, discrete mathematics, algebra",
    "Technology - cybersecurity, artificial intelligence, robotics, software engineering, quantum computing",
]

# Define the prompt for generating questions
prompt = """Given the following TOPIC, create a question that delves into a specific and focused aspect of the TOPIC with substantial depth and breadth. The question should be detailed, explore fundamental principles, inspire curiosity, and provoke deep thought.
TOPIC: {}"""

# Define various system messages to guide the AI's responses
MESSAGES = [
    "You should describe the task and provide a detailed explanation. For multiple choice questions, first identify the correct answer(s). Then, explain why the other options are incorrect. Explain as if to a five-year-old.",
    "You are an AI assistant. Describe the task and provide a detailed explanation. For multiple choice questions, first identify the correct answer(s). Then, explain why the other options are incorrect. You may need to use additional knowledge to answer.",
    "You are an AI assistant. Provide a comprehensive answer so the user does not need to look elsewhere for understanding.",
    "You are a helpful assistant, always providing clear explanations. Respond as if explaining to a five-year-old.",
    "You are a teacher. Given a task, explain in simple steps what is being asked, any provided guidelines, and how to use those guidelines to find the answer.",
    "You are an AI assistant that aids in information retrieval. Provide an in-depth answer so the user does not need to search elsewhere for clarity.",
    "You are an AI assistant. The user will give you a task. Your goal is to complete it as accurately as possible. Think through the steps and justify each one.",
    "You are an AI assistant. You will be given a task. Provide a thorough and detailed answer.",
    "Explain how you utilized the definition to arrive at the answer.",
    "The user will give you a task with specific instructions. Your job is to follow them meticulously. Think through the steps and justify your actions.",
]

# Number of questions to generate per topic
num_questions = 2
request_list = []

# Generate requests for questions based on topics
for topic in topics:
    request_list.extend(
        [
            {
                "messages": [
                    {"role": "system", "content": random.choice(MESSAGES)},
                    {"role": "user", "content": prompt.format(topic)},
                ]
            }
            for _ in range(num_questions)
        ]
    )

# Generate questions using the LLM
questions = llm.generate(request_list)
questions = [response[1]["choices"][0]["message"]["content"] for response in questions]

# Prepare requests for answers based on generated questions
request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in questions
]

# Generate answers using the LLM
answers = llm.generate(request_list)
answers = [response[1]["choices"][0]["message"]["content"] for response in answers]

# Save the questions and answers to a JSONL file
data = [{"Question": q, "Answer": a} for q, a in zip(questions, answers)]
with open('dataset.jsonl', 'w') as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
```

## Example Output
When you run the above code, it will generate a JSONL file named `dataset.jsonl` containing questions and their corresponding answers. The content of the file may look like this:

```json
{"Question": "What are the key factors that contribute to successful entrepreneurship in today's economy?", "Answer": "Successful entrepreneurship today relies on innovation, market research, and adaptability to changing consumer needs."}
{"Question": "How did the Industrial Revolution impact global trade?", "Answer": "The Industrial Revolution significantly increased production capacity, leading to a surge in global trade as countries sought raw materials and markets for their goods."}
```

