import time
import logging
import re
# Initialize logger
log = logging.getLogger(__name__)

class cot:
    
    def __init__(self,llm) -> None:
        self.llm = llm
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        
        
        cot_response_list = self.cot_reflection(request_list)
        model_response_list=[]
        for out_response in model_response_returned:
            model_response_list.append(out_response[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, cot_response in zip(model_response_list,cot_response_list):
            response.append({'model_response':model_response,"cot_response":cot_response})
        return response
    def gather_requests(self,request_list: list):
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
    def cot_reflection(self,request_list):
      responses_return=[]
      request_li_modified  = self.gather_requests(request_list)
      for request_in in range(len(request_list)):
          system_prompt = request_li_modified[request_in][0]
          initial_query = request_li_modified[request_in][1]
          cot_prompt = f"""
               {system_prompt}

               You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps properly:

               1. Think through the problem with step by step approach, within the <thinking> tags.
               2. Reflect on your thinking to check for any errors or improvements, within the <reflection> tags.
               3. Make any required adjustments based on your reflection.
               4. Provide your final, concise answer within the <output> tags.

               Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
               Do not include any part of the final answer in these sections. 
               The actual response to the query must be entirely contained within the <output> tags.

               Use the following format for your response:
               <thinking>
               [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
               <reflection>
               [Your reflection on your reasoning, checking for errors or improvements]
               </reflection>
               [Any adjustments to your thinking based on your reflection]
               </thinking>
               <output>
               [Your final, concise answer to the query. This is the only part that will be shown to the user.]
               </output>
               """
          messages=[
                  {"role": "system", "content": cot_prompt},
                  {"role": "user", "content": initial_query}
            ]
          request_list[request_in]['messages'] = messages

      # Make the API call
      response_list = self.llm.generate(request_list)

      # Extract the full response
      for response in response_list:
         full_response = response[1]['choices'][0]['message']['content']
       
         log.info(f"CoT with Reflection :\n{full_response}")

         # Use regex to extract the content within <thinking> and <output> tags
         thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
         output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)

         thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
         output = output_match.group(1).strip() if output_match else full_response

         log.info(f"Final output :\n{output}")

     
         responses_return.append(full_response)
      return responses_return
     