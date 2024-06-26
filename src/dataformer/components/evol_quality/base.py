import random
from typing import Any, Dict, List
from dataformer.llms.openllm import OpenLLM
from dataformer.components.evol_quality.prompts import MUTATION_TEMPLATES

class EvolQuality:

    def __init__(self, llm: OpenLLM, num_evolutions: int = 1, store_evolutions: bool = False, include_original_response: bool = False, mutation_templates: Dict[str, str] = MUTATION_TEMPLATES):
        self.llm = llm
        self.num_evolutions = num_evolutions
        self.store_evolutions = store_evolutions
        self.include_original_response = include_original_response
        self.mutation_templates = mutation_templates
        
    def evolve_responses(self, instruction: str, response: str) -> List[str]:
        """Evolves a response based on the mutation templates."""
        if self.include_original_response:
            evolved_responses = [response]
        else:
            evolved_responses = []
        for _ in range(self.num_evolutions):
            mutation = random.choice(list(self.mutation_templates.values()))
            mutated_prompt = mutation.replace("<PROMPT>", instruction).replace("<RESPONSE>", response)
            generation = self.llm.generate([{"model": self.llm.model,"messages": [{"role": "user", "content": mutated_prompt}]}])
            evolved_response = generation[0][2]['choices'][0]['message']['content']
            evolved_responses.append(evolved_response)
            response = evolved_response
        
        if not self.store_evolutions:
            return [evolved_responses[-1]]
        else:
            return evolved_responses

    def generate(self, inputs):
        """Generates evolved responses for a list of inputs containing instructions and responses."""
        self.results = []
        for input in inputs:
            instruction, response = input['instruction'], input['response']
            evolved_responses = self.evolve_responses(instruction, response)
            result = {
                "instruction": instruction,
                "response": response,
                "evolved_responses": evolved_responses
            }
            self.results.append(result)
        return self.results

