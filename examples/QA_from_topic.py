import random

import pandas as pd

from dataformer.llms.asyncllm import AsyncLLM

api_key = ""
llm = AsyncLLM(api_key, api_provider="openai")


topics = [
    "Science - physics, chemistry, biology, astronomy, etc.",
    "Mathematics - algebra, geometry, calculus, statistics, etc.",
    "Technology - computers, engineering, AI, robotics, etc.",
    "Business - economics, finance, marketing, management, entrepreneurship",
    "History - ancient, medieval, modern, world history, military history",
]

prompt = """For the following TOPIC, generate a question that covers a very narrow topic in the TOPIC, with sufficient depth and breadth. The topic in the question should be important to the TOPIC, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question. \n TOPIC: {}"""

MESSAGES = [
    "",
    "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant that helps people find information. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
    "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.",
    "Explain how you used the definition to come up with the answer.",
    "You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.",
    "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.",
    "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    "You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.",
    "Given a definition of a task and a sample input, break the definition into small parts. Each of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format:\n\nPart #: a key part of the definition.\nUsage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.",
    "You are an AI assistant that helps people find information.",
]

num_questions = 10
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

df = pd.DataFrame({"Question": questions, "Answer": answers})
df.to_csv("dataset.csv")
