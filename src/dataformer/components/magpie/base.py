import json

import pandas as pd
import requests
from datasets import Dataset
from tqdm import tqdm

from dataformer.components.base.prompts import languages, templates


class MAGPIE:
    def __init__(self, model, URL, lang="en"):
        self.model = model
        self.URL = URL
        self.lang = lang
        self.template = f"{templates[self.model]}{languages[self.lang]}:"

    def query(self, prompt, params=None, role="user"):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": role, "content": prompt}],
        }
        if not params:
            params = {"seed": 676, "temperature": 0.8, "top_p": 1}
        data = data | params

        response = requests.post(self.URL, json=data, headers=headers)
        response = json.loads(response.content.decode("utf-8"))

        answer = response["message"]["content"]
        return answer

    def extract(self, text):
        for content in text.split("\n"):
            if content:
                return content.strip()

    def validate(self, entry):
        if entry["instruction"] is None or entry["response"].strip() == "":
            return False
        return True

    def display(self, num_samples):
        print("Creating dataset with the following parameters:")
        print(f"MODEL: {self.model}")
        print(f"Total Samples: {num_samples}")
        print("Language: English")
        print(f"Query Template: {self.template}")

    def generate(self, num_samples, params=None, verbose=False):
        self.display(num_samples)
        dataset = []
        valid_samples = 0
        with tqdm(total=num_samples, desc="\n Generating Samples") as pbar:
            while valid_samples < num_samples:
                result = self.query(self.template, params, "assistant")
                instruction = self.extract(result)
                response = self.query(instruction, params)

                entry = {
                    "instruction": instruction,
                    "response": response,
                }
                if self.validate(entry):
                    dataset.append(entry)
                    if verbose:
                        print(f" --- Instruction: {instruction}")
                        print(f" --- Response: {response}")
                    valid_samples += 1
                    pbar.update(1)
        return dataset

    def _save_dataset(self, dataset, filepath, hf_token=None):
        name, format = filepath.split(".")
        df = pd.DataFrame(dataset)

        if hf_token:
            dataset = Dataset.from_pandas(df)
            dataset.push_to_hub(name, token=hf_token, private=True)

        if format == "csv":
            df.to_csv(self.filepath)
        elif format == "xlsx":
            df.to_excel(self.filepath)
        elif format == "json":
            df.to_json(self.filepath, orient="records")
