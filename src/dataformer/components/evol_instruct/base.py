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
        # Prepare all mutated instructions first
        self.mutated_instructions = []
        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_instruction = mutation.replace("<PROMPT>", instruction)
            self.mutated_instructions.append(mutated_instruction)

        # Generate all evolved instructions in one call
        responses = self.llm.generate([{"model": self.llm.model, "messages": [{"role": "user", "content": mi}]} for mi in self.mutated_instructions])
        
        # Extract and store the evolved instructions
        for response in responses:
            evolved_instructions.append(response[2]['choices'][0]['message']['content'])

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
                answer = self.llm.generate([{"model": self.llm.model,"messages": [{"role": "user", "content": evolved_instruction}]} for evolved_instruction in evolved])
                self.answers = [ans[2]['choices'][0]['message']['content'] for ans in answer]
                result["answers"] = self.answers
            self.results.append(result)
        return self.results    
