import logging
from typing import List, Dict
from difflib import SequenceMatcher

# Initialize logger
log = logging.getLogger(__name__)
class SelfConsistency:
    def __init__(self,llm,num_samples=5,similarity_threshold=0.8) -> None:
        self.llm = llm
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        model_response=[]

        
        selfconsistency_returned_response_list = self.return_final_answer(request_list=request_list)
        log.info(selfconsistency_returned_response_list)
        selfconsistency_response_list=[]
        for answer in selfconsistency_returned_response_list:
            if answer['aggregated_result']['clusters']:
                    selfconsistency_response_list.append(answer['aggregated_result']['clusters'][0]['answer'])
            else:
                    selfconsistency_response_list.append("No consistent answer found.")
        model_response_list=[]
        for out_response in model_response_returned:
            model_response_list.append(out_response[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, selfconsistency_response in zip(model_response_list,selfconsistency_response_list):
            response.append({'model_response':model_response,"SelfConsistency_response":selfconsistency_response})
        return response
    

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
    
    def generate_responses(self, request_list):
        
        request_list_modified  =self.gather_requests(request_list)

        responses_return = []
        for i in range(len(request_list)):
            system_prompt = request_list_modified[i][0]
            user_prompt = request_list_modified[i][1]
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            request_list[i]['messages'] = messages
        
            responses_li=[]
            responses = self.llm.generate([request_list[i].copy() for _ in range(self.num_samples)])
            for response_output in responses:    
                responses_li.append(response_output[1]['choices'][0]['message']['content'])
            responses_return.append(responses_li)
        return responses_return

    def calculate_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def cluster_similar_responses(self, responses):
        clusters = []
        for response in responses:
            added_to_cluster = False
            for cluster in clusters:
                if self.calculate_similarity(response, cluster[0]) >= self.similarity_threshold:
                    cluster.append(response)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
        return clusters

    def aggregate_results(self, responses) -> Dict[str, any]:
        final_answers = responses
        cluster_answer=[]
        return_final_answer=[]
        for answer in final_answers:
            clusters = self.cluster_similar_responses(answer)
        
            cluster_info = []
            for cluster in clusters:
                cluster_info.append({
                    "answer": cluster[0],
                    "frequency": len(cluster),
                    "variants": cluster
                })
            
            cluster_info.sort(key=lambda x: x['frequency'], reverse=True)
            
            cluster_answer.append({
                "clusters": cluster_info,
                "total_responses": len(answer),
                "num_unique_clusters": len(clusters)
            })
        return cluster_answer
    
    def return_final_answer(self,request_list):
        answer_response_list = self.generate_responses(request_list)
        result_clusters = self.aggregate_results(answer_response_list)
        final_answer=[]
        for i in range(len(answer_response_list)):
            final_answer.append({
            "individual_responses": answer_response_list[i],
            "aggregated_result": result_clusters[i]
        })
        return final_answer
                       

