# Create QAs from a Topic

import random

from dataformer.llms.asyncllm import AsyncLLM
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.environ.get("OPENAI_API_KEY") 
llm = AsyncLLM(api_key=api_key, api_provider="openai")


topics = [
    "Business and Economics - entrepreneurship, economic theory, investment strategies, marketing, supply chain management",
    "Historical Studies - industrial revolution, global history, military history, ancient civilizations, renaissance",
    "Natural Sciences - botany, chemistry, astronomy, marine biology, physics",
    "Mathematics - number theory, statistics, calculus, discrete mathematics, algebra",
    "Technology - cybersecurity, artificial intelligence, robotics, software engineering, quantum computing",
]

prompt = """Given the following TOPIC, create a question that delves into a specific and focused aspect of the TOPIC with substantial depth and breadth.The question should be detailed, explore fundamental principles, inspire curiosity, and provoke deep thought.
TOPIC: {}"""


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
    "",
]

num_questions = 2
request_list = []

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

questions = llm.generate(request_list)
questions = [response[1]["choices"][0]["message"]["content"] for response in questions]

request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in questions
]
answers = llm.generate(request_list)
answers = [response[1]["choices"][0]["message"]["content"] for response in answers]

import json

data = [{"Question": q, "Answer": a} for q, a in zip(questions, answers)]
with open('dataset.jsonl', 'w') as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')