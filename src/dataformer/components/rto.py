import time
import re
import logging

# Initialize logger
log = logging.getLogger(__name__)


class rto:
    def __init__(self,llm) -> None:
        self.llm = llm
        
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        
        
        rto_response_list = self.round_trip_optimization(request_list)
        model_response_list=[]
        for out_response in model_response_returned:
            model_response_list.append(out_response[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, rto_response in zip(model_response_list,rto_response_list):
            response.append({'model_response':model_response,"rto_response":rto_response})
        return response


    def extract_code(self,text_content: str):
        # Define regex to extract code given by model between triple backticks
        code_block_pattern = r"```(.*?)```"
        
        # Attempt to find code block
        match = re.search(code_block_pattern, text_content, re.DOTALL)
        
        # If code block found, return it after stripping whitespace
        if match:
            return match.group(1).strip()
        else:
            log.warning("Failed to extract the code block. Returning the original text.")
            return text_content
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

    def round_trip_optimization(self, request_list: list) -> list:
        request_list_modified = self.gather_requests(request_list)
        response_list=[]

        request_c1_list =[]
        for request_m,request in zip(request_list_modified,request_list):
            messages=[]
            system_prompt = request_m[0]
            initial_query = request_m[1]
           
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}]
            request['messages'] = messages
            request_c1_list.append(request)
        # print("request_c1_list",request_c1_list)
        # Generate initial code (C1)
        response_c1_list = self.llm.generate(request_c1_list)
        # print("response_c1_list",response_c1_list)
        c1_list=[]
        for response_c1_index in range(len(response_c1_list)):
            
            c1 = response_c1_list[response_c1_index][1]['choices'][0]['message']['content']
            c1_list .append(c1)    
            # Generate description of the code (Q2)
            
            request_c1_list[response_c1_index]['messages'].append({"role": "assistant", "content": c1})
            request_c1_list[response_c1_index]['messages'].append({"role": "user", "content": "Summarize or describe the code given to you. \
                             Ensure, that the summary should be in such form of instruction that, given the same instruction you can create the code by yourself."})
        # print("request_c1_list",request_c1_list)
        response_q2_list =self.llm.generate(request_c1_list)
        # print("response_q2_list",response_q2_list)
        request_c2_list=[]
        for response_q2_index in range(len(response_q2_list)):
            q2 = response_q2_list[response_q2_index][1]['choices'][0]['message']['content']
            messages=[]
            # Generate second code based on the description (C2)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": q2}]
            request = request_list[response_q2_index]
            request['messages'] = messages
            request_c2_list.append(request)
        # print("request_c2_list",request_c2_list)
        response_c2_list = self.llm.generate(request_c2_list)
        c2_list=[]
        for response_c2 in response_c2_list:
            c2_list.append(response_c2[1]['choices'][0]['message']['content'])


        request_c3_list=[]
        for c1,c2,request_m,request in zip(c1_list,c2_list,request_list_modified,request_list) :
            c1 = self.extract_code(c1)
            c2 = self.extract_code(c2)
            system_prompt = request_m[0]
            initial_query = request_m[1]
            

            if c1.strip() == c2.strip():
                return c1
            messages=[]
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Initial query: {initial_query}\n\nFirst generated code (C1):\n{c1}\n\nSecond generated code (C2):\n{c2}\n\nBased on the initial query and these two different code implementations, generate a final, optimized version of the code. Only respond with the final code, do not return anything else."}]
            request['messages'] = messages
            request_c3_list.append(request)
        # print("request_c3_list",request_c3_list)
        response_c3_list =self.llm.generate(request_c3_list)
        # print("response_c3_list",response_c3_list)
        for response_c3 in response_c3_list:

            c3 = response_c3[1]['choices'][0]['message']['content']
            response_list.append(response_c3[1]['choices'][0]['message']['content'])
        # print(response_list)
        return response_list
    

