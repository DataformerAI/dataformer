import random
from typing import Any, Dict, List

from dataformer.components.evol_instruct.prompts import MUTATION_TEMPLATES
from dataformer.llms import AsyncLLM
from dataformer.utils.cache import (
    get_cache_vars,
    get_request_details,
    task_id_generator_function,
)


class EvolInstruct:
    def __init__(
        self,
        llm: AsyncLLM,
        num_evolutions: int = 1,
        store_evolutions: bool = False,
        generate_answers: bool = False,
        include_original_instruction: bool = False,
        mutation_templates: Dict[str, str] = MUTATION_TEMPLATES,
    ):
        self.llm = llm
        self.num_evolutions = num_evolutions
        self.store_evolutions = store_evolutions
        self.generate_answers = generate_answers
        self.include_original_instruction = include_original_instruction
        self.mutation_templates = mutation_templates
        self.use_cache = True
        self.active_id = 0

    def evolve_instruction(self, instruction: str) -> List[str]:
        """Evolves an instruction based on the mutation templates."""
        if self.include_original_instruction:
            evolved_instructions = [instruction]
        else:
            evolved_instructions = []

        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_instruction = mutation.replace("<PROMPT>", instruction)
            response = self.llm.generate(
                [
                    {
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": mutated_instruction}],
                    }
                ],
                use_cache=self.use_cache,
                cache_vars=self.cache_vars,
                task_id_generator=self.task_id_generator,
            )
            evolved_instruction = response[self.active_id][1]["choices"][0]["message"]["content"]
            evolved_instructions.append(evolved_instruction)
            instruction = evolved_instruction
            if self.use_cache:
                self.active_id += 1

        if not self.store_evolutions:
            return [evolved_instructions[-1]]
        return evolved_instructions

    def generate(self, instructions, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Generates evolved instructions for a list of instructions."""

        self.use_cache = use_cache
        self.active_id = 0

        self.task_id_generator = task_id_generator_function()

        self.request_details = get_request_details(
            [
                {
                    "model": self.llm.model,
                    "messages": [{"role": "user", "content": instruction}],
                }
                for instruction in instructions
            ]
        )

        # Get cache_vars after initalizing all important variables
        self.cache_vars = get_cache_vars(
            self,
            ignore_keys=["results", "task_id_generator", "cache_vars", "use_cache", "active_id"],
        )

        self.results = []
        all_messages = []
        for instruction in instructions:
            evolved = self.evolve_instruction(instruction)
            result = {
                "original_instruction": instruction,
                "evolved_instructions": evolved,
            }
            if self.generate_answers:
                # Prepare all messages for batch processing
                all_messages.extend(
                    [
                        {
                            "model": self.llm.model,
                            "messages": [{"role": "user", "content": ei}],
                        }
                        for ei in evolved
                    ]
                )
            self.results.append(result)

        # Perform a single batch request for all messages
        if self.generate_answers:

            self.request_details = get_request_details(
                [
                    {
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": str(instruction)}],
                    }
                    for instruction in all_messages
                ]
            )

            self.cache_vars = get_cache_vars(
                self,
                ignore_keys=["results", "task_id_generator", "cache_vars", "use_cache", "active_id"],
            )

            answers = self.llm.generate(
                all_messages, use_cache=self.use_cache, cache_vars=self.cache_vars
            )
            answer_index = 0
            if self.include_original_instruction:
                num_answers = self.num_evolutions + 1
            else:
                num_answers = self.num_evolutions
            for result in self.results:
                result["answers"] = [
                    answers[i][1]["choices"][0]["message"]["content"]
                    for i in range(answer_index, answer_index + num_answers)
                ]
                answer_index += num_answers

        return self.results
