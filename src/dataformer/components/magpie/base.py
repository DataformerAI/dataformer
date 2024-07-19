from dataformer.components.magpie.prompts import languages, templates


class MAGPIE:
    def __init__(self, llm, lang="en"):
        self.llm = llm
        self.model = self.llm.model
        self.lang = lang
        self.template = f"{templates[self.model]}{languages[self.lang]}:"

    def create_requests(self, prompt, params=None, role="user"):
        data = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": role, "content": prompt}],
        }
        if not params:
            params = {"seed": 676, "temperature": 0.8, "top_p": 1}
        request = data | params
        return request

    def extract(self, text):
        for content in text.split("\n"):
            if content:
                return content.strip()

    def validate(self, entry):
        if entry["question"] is None or entry["answer"].strip() == "":
            return False
        return entry

    def display(self, num_samples):
        print("Creating dataset with the following parameters:")
        print(f"MODEL: {self.model}")
        print(f"Total Samples: {num_samples}")
        print("Language: English")
        print(f"Query Template: {self.template}")

    def generate(self, num_samples, params=None, use_cache=False):
        self.display(num_samples)

        request_list = [self.create_requests(self.template, params, "assistant") for _ in range(num_samples)]
        response_list = self.llm.generate(request_list, use_cache=use_cache)

        questions = [response[1]["choices"][0]["message"]["content"] for response in response_list]
        questions = list(filter(self.extract, questions))

        request_list = [self.create_requests(question) for question in questions]
        response_list = self.llm.generate(request_list, use_cache=use_cache)

        answers = [response[1]["choices"][0]["message"]["content"] for response in response_list]

        dataset = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

        dataset = list(filter(self.validate, dataset))
        return dataset
