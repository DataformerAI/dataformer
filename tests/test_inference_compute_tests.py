from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dataformer.llms import AsyncLLM
from dataformer.components.SelfConsistency import SelfConsistency
from dataformer.components.cot import cot
from dataformer.components.pvg import pvg
from dataformer.components.rto import rto
llm = AsyncLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_provider="deepinfra"
)  
def test_cot():
    request_list = [
        {"messages": [{"role": "user", "content": "If a train leaves a station traveling at 60 miles per hour and another train leaves the same station 30 minutes later traveling at 90 miles per hour, when will the second train catch up to the first train?"}]}
    ]
   
    cot_instance = cot(llm)
    results = cot_instance.generate(request_list)
    assert results is not None, "Response is missing"
    assert results[0]['cot_response'] is not None, "No CoT response"
    assert results[0]['model_response'] is not None, "No model response"

def test_self_consistency():
    request_list = [
        {"messages": [{"role": "user", "content": "I have a dish of potatoes. The following statements are true: No potatoes of mine, that are new, have been boiled. All my potatoes in this dish are fit to eat. No unboiled potatoes of mine are fit to eat. Are there any new potatoes in this dish?"}]}
    ]
   
    self_consistency_instance = SelfConsistency(llm)
    results = self_consistency_instance.generate(request_list=request_list, return_model_answer=True)
    assert results is not None, "Response is missing"
    assert results[0]['SelfConsistency_response'] is not None, "No SelfConsistency response"
    assert results[0]['model_response'] is not None, "No model response"

def test_pvg():
    request_list = [
        {"messages": [{"role": "user", "content": "Write a code in python for timetable generation. Consider all the constraints."}]}
    ]
    
    pvg_instance = pvg(llm)
    results = pvg_instance.generate(request_list)
    assert results is not None, "Response is missing"
    assert results[0]['pvg_response'] is not None, "No PVG response"
    assert results[0]['model_response'] is not None, "No model response"

def test_rto():
    request_list = [
        {"messages": [{"role": "user", "content": "Write a genetic algorithm code in python which is fast."}]}
    ]
    rto_instance = rto(llm)
    results = rto_instance.generate(request_list)
    assert results is not None, "Response is missing"
    assert results[0]['rto_response'] is not None, "No RTO response"
    assert results[0]['model_response'] is not None, "No model response"
def test_inference_compute_tests():
    test_cot()
    test_self_consistency()
    test_pvg()
    test_rto()
