import random
from typing import Dict, List

from dataformer.components.evol_quality.prompts import MUTATION_TEMPLATES
from dataformer.llms import AsyncLLM
from dataformer.utils.cache import (
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
)


class EvolQuality:
    def __init__(
        self,
        llm: AsyncLLM,
        num_evolutions: int = 1,
        store_evolutions: bool = False,
        include_original_response: bool = False,
        mutation_templates: Dict[str, str] = MUTATION_TEMPLATES,
    ):
        self.llm = llm
        self.num_evolutions = num_evolutions
        self.store_evolutions = store_evolutions
        self.include_original_response = include_original_response
        self.mutation_templates = mutation_templates
        self.use_cache = True
        self.active_id = 0

    def evolve_responses(self, instruction: str, response: str) -> List[str]:
        """Evolves a response based on the mutation templates."""
        if self.include_original_response:
            evolved_responses = [response]
        else:
            evolved_responses = []
        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_prompt = mutation.replace("<PROMPT>", instruction).replace(
                "<RESPONSE>", response
            )
            generation = self.llm.generate(
                [
                    {
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": mutated_prompt}],
                    }
                ],
                use_cache=self.use_cache,
                cache_vars=self.cache_vars,
                task_id_generator=self.task_id_generator,
            )
            evolved_response = generation[self.active_id][1]["choices"][0]["message"]["content"] # Response is list of all responses (including prev inputs).
            evolved_responses.append(evolved_response)
            response = evolved_response
            if self.use_cache:
                self.active_id += 1

        if not self.store_evolutions:
            return [evolved_responses[-1]]
        else:
            return evolved_responses

    def generate(self, inputs, use_cache: bool = True):
        """Generates evolved responses for a list of inputs containing instructions and responses."""

        self.use_cache = use_cache
        self.active_id = 0

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
            ignore_keys=["results", "task_id_generator", "cache_vars", "use_cache", "active_id"],
        )

        self.results = []
        for input in inputs:
            instruction, response = input["instruction"], input["response"]
            evolved_responses = self.evolve_responses(instruction, response)
            result = {
                "instruction": instruction,
                "response": response,
                "evolved_responses": evolved_responses,
            }
            self.results.append(result)
        return self.results
