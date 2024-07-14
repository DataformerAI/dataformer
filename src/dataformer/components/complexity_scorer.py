import re
from typing import List, Dict, Any, Union
from jinja2 import Template
from dataformer.llms import AsyncLLM
from dataformer.utils.cache import (
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
)
import sys
if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources



_PARSE_SCORE_LINE_REGEX = re.compile(r"\[\d+\] Score: (\d+)", re.IGNORECASE)

class ComplexityScorer:
    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self.template = self._load_template()
        self.use_cache = True
        

    def _load_template(self) -> Template:
        # Load the Jinja2 template
        _path = str(
            importlib_resources.files("dataformer")
            / "components"
            / "templates"
            / "complexity-scorer.jinja2"
        )
        return Template(open(_path).read())

    def _parse_scores(self, output: Union[str, None], input: Dict[str, Any]) -> List[float]:
        if output is None:
            return {"scores": [None] * len(input["instructions"])}
        scores = []
        score_lines = output.split("\n")
        for i, line in enumerate(score_lines):
            match = _PARSE_SCORE_LINE_REGEX.match(line)
            score = float(match.group(1)) if match else None
            scores.append(score)
            if i == len(input["instructions"]) - 1:
                break
        return scores

    def score(self, inputs: List[Dict[str, Any]], use_cache: bool = True) -> List[Dict[str, Any]]:
        
        self.use_cache = use_cache
        self.task_id_generator = task_id_generator_function()
        self.request_details = get_request_details(
            [
                {
                    "model": self.llm.model,
                    "messages": [{"role": "user", "content": inputdata}],
                }
                for inputdata in inputs
            ]
        )
        # Get cache_vars after initalizing all important variables
        self.cache_vars = get_cache_vars(
            self,
            ignore_keys=["results", "task_id_generator", "cache_vars", "use_cache"],
        )
        
        self.results = []
        requests = []
        for input in inputs:
            prompt = self.template.render(instructions=input['instructions'])
            requests.append({
                            "model": self.llm.model,
                            "messages": [{"role": "user", "content": prompt}],
                        })
            result = {
                "instructions": input['instructions'],
            }
            self.results.append(result)

        responses = self.llm.generate(requests,
                                      use_cache=self.use_cache,
                                      cache_vars=self.cache_vars,
                                      task_id_generator=self.task_id_generator)
        for response, input, result  in zip(responses, inputs, self.results):
            result['scores'] = self._parse_scores(response[1]['choices'][0]['message']['content'], input)
            result["raw output"] = response[1]['choices'][0]['message']['content']

        return self.results