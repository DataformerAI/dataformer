from dataformer.components.magpie.prompts import languages, templates


class MAGPIE:
    def __init__(self, llm, template=None, lang="en"):
        self.llm = llm
        self.lang = lang

        if template:
            self.template = template

        else:
            if "llama3" or "llama-3" in self.llm.model:
                self.template = templates["llama3"]

        self.template = f"{self.template}{languages[self.lang]}:"

    def create_requests(self, prompt, role="user"):
        data = {
            "model": self.llm.model,
            "stream": False,
            "messages": [{"role": role, "content": prompt}],
        }
        return data

    def extract(self, text):
        for content in text.split("\n"):
            if content:
                return content.strip()
        return text.strip()

    def validate(self, entry):
        if entry["question"] is None or entry["answer"].strip() == "":
            return False
        return entry

    def display(self, num_samples):
        print("Creating dataset with the following parameters:")
        print(f"MODEL: {self.llm.model}")
        print(f"Total Samples: {num_samples}")
        print("Language: English")
        print(f"Query Template: {self.template}")

    def generate(self, num_samples, use_cache=False):
        self.display(num_samples)

        request_list = [self.create_requests(self.template, "assistant") for _ in range(num_samples)]
        response_list = self.llm.generate(request_list, use_cache=use_cache)

        questions = [response[1]["choices"][0]["message"]["content"] for response in response_list]
        questions = list(filter(self.extract, questions))

        request_list = [self.create_requests(question) for question in questions]
        response_list = self.llm.generate(request_list, use_cache=use_cache)

        answers = [response[1]["choices"][0]["message"]["content"] for response in response_list]

        dataset = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        dataset = list(filter(self.validate, dataset))

        return dataset
