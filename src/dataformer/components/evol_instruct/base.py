import random
from typing import Any, Dict, List

from dataformer.components.evol_instruct.prompts import MUTATION_TEMPLATES
from dataformer.llms.openllm import OpenLLM


class EvolInstruct:

    def __init__(self, llm: OpenLLM, num_evolutions: int = 1, store_evolutions: bool = False, generate_answers: bool = False, include_original_instruction: bool = False, mutation_templates: Dict[str, str] = MUTATION_TEMPLATES):
        self.llm = llm
        self.num_evolutions = num_evolutions
        self.store_evolutions = store_evolutions
        self.generate_answers = generate_answers
        self.include_original_instruction = include_original_instruction
        self.mutation_templates = mutation_templates

    def evolve_instruction(self, instruction: str) -> List[str]:
        """Evolves an instruction based on the mutation templates."""
        if self.include_original_instruction:
            evolved_instructions = [instruction]
        else:
            evolved_instructions = []

        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_instruction = mutation.replace("<PROMPT>", instruction)
            response = self.llm.generate([{"model": self.llm.model,"messages": [{"role": "user", "content": mutated_instruction}]}])
            evolved_instruction = response[0][2]['choices'][0]['message']['content']
            evolved_instructions.append(evolved_instruction)
            instruction = evolved_instruction

        if not self.store_evolutions:
            return [evolved_instructions[-1]]
        return evolved_instructions

    def generate(self, instructions) -> List[Dict[str, Any]]:
        """Generates evolved instructions for a list of instructions."""
        self.results = []
        all_messages = []
        for instruction in instructions:
            evolved = self.evolve_instruction(instruction)
            result = {
                "original_instruction": instruction,
                "evolved_instructions": evolved
            }
            if self.generate_answers:
                # Prepare all messages for batch processing
                all_messages.extend([{"model": self.llm.model, "messages": [{"role": "user", "content": ei}]} for ei in evolved])
            self.results.append(result)

        # Perform a single batch request for all messages
        if self.generate_answers:
            answers = self.llm.generate(all_messages)
            answer_index = 0
            if self.include_original_instruction:
                num_answers = self.num_evolutions+1
            else:
                num_answers = self.num_evolutions
            for result in self.results:
                result["answers"] = [answers[i][2]['choices'][0]['message']['content'] for i in range(answer_index, answer_index + num_answers)]
                answer_index += num_answers

        return self.results
