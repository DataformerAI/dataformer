import time
from typing import List, Tuple
import logging
import re
logger = logging.getLogger(__name__)
class pvg:
    def __init__(self,llm,num_rounds: int = 3, num_solutions: int = 2,verify_model="meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        self.llm = llm
        self.num_rounds =num_rounds
        self.num_solutions= num_solutions
        self.verify_model = verify_model
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        
        
        pvg_response_list = self.pvg(request_list)
        model_response_list=[]
        for out_response in model_response_returned:
            model_response_list.append(out_response[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, pvg_response in zip(model_response_list,pvg_response_list):
            response.append({'model_response':model_response,"pvg_response":pvg_response})
        return response
        

    def generate_solutions(self, request_list, request_list_modified, num_solutions: int, is_sneaky: bool = False, temperature: float = 0.7) :
        for i in range(len(request_list)):
            role = "sneaky" if is_sneaky else "helpful"
            logger.info(f"Generating {num_solutions} {role} solutions")
            system_prompt=request_list_modified[i][0]
            query = request_list_modified[i][1]
            role_instruction = """
            You are a problem solver, given with the task of generating solutions to the given problem. 
            If you are in 'helpful' mode:
            - Provide a correct, detailed, and step-by-step solution to the problem.
            - Explain your reasoning clearly and logically.
            - Ensure your final answer or conclusion is perfectly accurate and well-supported.
            - Use domain-specific terminology or notation is applicable or appropriate.

            If you are in 'sneaky' mode:
            - Provide a solution that appears correct but contains subtle errors or flaws.
            - Make your explanation seem logical and convincing.
            - Ensure your final answer or conclusion is incorrect, but in a way that's not immediately obvious.
            - Maintain the appearance of expertise in the relevant domain.
            """
            
            messages = [
                {"role": "system", "content": f"{system_prompt}\n{role_instruction}\nYou are in {role} mode."},
                {"role": "user", "content": query}
            ]

            request_list[i]['messages'] = messages
            request_list[i]['temperature'] = temperature
            request_list[i]['n'] = num_solutions
        responses_list = self.llm.generate(request_list)
        solutions_list=[]
        for response in responses_list:
            solutions = [choice['message']['content'] for choice in response[1]['choices']]
            solutions_list.append(solutions)
        # logger.debug(f"Generated {role} solutions: {solutions_list}")
        return solutions_list

    def verify_solutions(self, system_prompt, initial_query, solutions):
          
            logger.info(f"Verifying {len(solutions)} solutions")
            verify_prompt = f"""{system_prompt}
                You are a verifier tasked with evaluating the correctness and clarity of solutions to the given problem.
                Rate the following solution on a scale from 0 to 10, where:
                - 0 is completely incorrect or incomprehensible
                - 5 is partially correct or lacks clarity
                - 10 is perfectly correct, clear, and well-explained

                Consider the following criteria:
                1. Accuracy of the final answer or conclusion
                2. Correctness of each step or argument in the solution
                3. Clarity and logical flow of the explanation
                4. Appropriate use of domain-specific concepts or terminology
                5. Completeness of the solution

                Be especially vigilant for subtle errors or flaws that might not be immediately obvious.

                Provide your response in the following format:

                Score: [Your numerical score from 0 to 10]
                Explanation: [Your detailed explanation for the score, highlighting specific strengths or weaknesses]

                Ensure that the Score is a single number between 0 and 10, and the Explanation is on a new line."""

            scores = []
            request_list=[]
            for i, solution in enumerate(solutions):
                messages = [
                    {"role": "system", "content": verify_prompt},
                    {"role": "user", "content": f"Problem: {initial_query}\n\nSolution: {solution}"}
                ]
                request_list.append({"messages":messages,'temperature':0.2})
            response_list = self.llm.generate(request_list)
            for response in response_list:
                rating = response[1]['choices'][0]['message']['content']
                logger.debug(f"Raw rating for solution {i+1}: {rating}")

                score_match = re.search(r"Score:\s*(\d+(\.\d+)?)", rating)
                explanation_match = re.search(r"Explanation:\s*(.*)", rating, re.DOTALL)

                if score_match:
                    try:
                        score = float(score_match.group(1))
                        scores.append(score)
                        # logger.debug(f"Solution {i+1} score: {score}")
                        # if explanation_match:
                            # explanation = explanation_match.group(1).strip()
                            # logger.debug(f"Explanation: {explanation}")
                        # else:
                            # logger.warning(f"No explanation found for solution {i+1}")
                    except ValueError:
                        scores.append(0)
                        # logger.warning(f"Failed to parse score for solution {i+1}. Setting score to 0.")
                else:
                    scores.append(0)
                    # logger.warning(f"No score found for solution {i+1}. Setting score to 0.")

            return scores

    def gather_requests(self,request_list):
        request_list_return =[]
        for request in request_list:
            initial_query=""
            system_prompt = ""
            conversation = []
    
            for message in request['messages']:
                role = message['role']
                content = message['content']
                
                if role == 'system':
                    system_prompt = content
                elif role in ['user', 'assistant']:
                    conversation.append(f"{role.capitalize()}: {content}")
            
            initial_query = "\n".join(conversation)
            request_list_return.append([system_prompt,initial_query])
        return request_list_return

    
    def pvg(self,request_list):
        logger.info(f"Starting inference-time PV game with {self.num_rounds} rounds and {self.num_solutions} solutions per round")
    

        request_list_modified = self.gather_requests(request_list)
        best_solution_list= ["" for i in range(len(request_list))]
        
        
        best_score_list = [-1 for i in range(len(request_list))]
        for round in range(self.num_rounds):
            logger.info(f"Starting round {round + 1}")
            
            temperature = max(0.2, 0.7 - (round * 0.1))
        
            helpful_solutions = self.generate_solutions(request_list, request_list_modified, self.num_solutions, temperature=temperature)
            sneaky_solutions = self.generate_solutions(request_list, request_list_modified, self.num_solutions, is_sneaky=True, temperature=temperature)
            request_list_refine=[]
            for i in range(len(helpful_solutions)):
                
                all_solutions = helpful_solutions[i] + sneaky_solutions[i]
                system_prompt = request_list_modified[i][0]
                initial_query = request_list_modified[i][1]
                best_score=best_score_list[i]
                scores = self.verify_solutions(system_prompt, initial_query, all_solutions)

                round_best_solution = max(zip(all_solutions, scores), key=lambda x: x[1])
                
                if round_best_solution[1] > best_score:
                    best_solution_list[i] = round_best_solution[0]
                    best_score_list[i] = round_best_solution[1]
                    logger.info(f"New best solution found in round {round + 1} with score {best_score}")
                else:
                    logger.debug(f"No improvement in round {round + 1}. Best score remains {best_score}")
                    
            if round < self.num_rounds - 1:
                    for j in range(len(request_list_modified)):
                        system_prompt= request_list_modified[j][0]
                        initial_query = request_list_modified[j][1]
                        logger.debug("Refining query for next round")
                        refine_prompt = f"""
                        Based on the original query and the best solution so far, suggest a refined query that might lead to an even better solution.
                        Focus on aspects of the problem that were challenging or not fully addressed in the best solution.
                        Maintain the core intent of the original query while adding specificity or context that could improve the solution.
                        
                        Original query: {initial_query}
                        
                        Best solution so far: {best_solution_list[j]}
                        
                        Refined query:
                        """
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": refine_prompt}
                        ]
                        request_list_refine.append({"messages":messages,"temperature":0.5})

           
                    response_listt = self.llm.generate(request_list_refine)
                    print(len(response_listt),len(request_list_modified))
                    for k in range(len(request_list_modified)):
                        request_list_modified[k][1] = response_listt[k][1]['choices'][0]['message']['content']

        return best_solution_list
