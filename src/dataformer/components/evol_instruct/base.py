import random
from typing import Any, Dict, List
from dataformer.llms.openllm import OpenLLM
from dataformer.components.evol_instruct.prompts import MUTATION_TEMPLATES

class Evol_Instruct:
    
    def __init__(self, llm: OpenLLM, num_evolutions: int, store_evolutions: bool = False, generate_answers: bool = False, include_original_instruction: bool = False, mutation_templates: Dict[str, str] = MUTATION_TEMPLATES):
        self.llm = llm
        self.num_evolutions = num_evolutions
        self.store_evolutions = store_evolutions
        self.generate_answers = generate_answers
        self.include_original_instruction = include_original_instruction
        self.mutation_templates = mutation_templates

    def evolve_instruction(self, instruction: str) -> List[str]:
        """Evolves an instruction based on the mutation templates."""
        evolved_instructions = [instruction]
        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_instruction = mutation.replace("<PROMPT>", instruction)
            # Generate evolved instruction using the LLM
            response = self.llm.generate([{"model": self.llm.model,"messages": [{"role": "user", "content": mutated_instruction}]}])
            evolved_instruction = response[0][2]['choices'][0]['message']['content']  # Assuming response is a list of strings
            evolved_instructions.append(evolved_instruction)
            instruction = evolved_instruction  # Update instruction for next iteration

        if not self.store_evolutions:
            return [evolved_instructions[-1]]
        return evolved_instructions

    def generate(self, instructions) -> List[Dict[str, Any]]:
        """Generates evolved instructions for a list of instructions."""
        self.results = []
        for instruction in instructions:
            evolved = self.evolve_instruction(instruction)
            result = {
                "original_instruction": instruction,
                "evolved_instructions": evolved
            }
            if self.generate_answers:
                self.answers = []
                for evolved_instruction in evolved:
                    answer = self.llm.generate([{"model": self.llm.model,"messages": [{"role": "user", "content": evolved_instruction}]}])
                    self.answers.append(answer[0][2]['choices'][0]['message']['content'])
                result["answers"] = self.answers
            self.results.append(result)
        return self.results    
