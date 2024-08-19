# Test Files Description

The folder contains **test files created on combinations of different parameters and methods**.

To Run all the tests together, run the following commands:

1. Go to the test folder
```
> cd tests
```

2. Run all the tests with the following command
```
> pytest
```

To Run tests individually, call the function in each file and run individual files by following commands.
```
> cd tests
> python test_filename.py
```
or 
```
> python tests\test_filename.py
```
## test_all_api_providers_in_request_list.py
- Purpose: This test is created to verify if all supported api_providers work perfectly and ensure smooth working when all requests contain an api provider, model and relevant api_keys. Test if api_provider and api_key are either overriding or assigned properly  with requests_list.


## test_cache_vars.py
- Purpose: This test is created to check if additional cache vars are taken into consideration for caching.

## test_caching_requests.py:
- Purpose: This test is created to check if cache creation, skipping and deletion, works perfectly with different requests and projects. It deletes specific project caches, multiple project caches, and clearing the entire cache and checks if the files are deleted as well as their associations are deleted.

## test_complexity_scorer.py
- Purpose: This test is created to verify the functionality of the `complexity scorer` component. It checks if the complexity scorer works perfectly by providing specific instructions and validating the scores and raw output.

## test_deita.py
- Purpose: This test is created to check if the `Deita` component works perfectly.

## test_embeddings.py
- Purpose: This test is created to verify the proper working of embedding models by making requests to embedding models URLs. 

## test_eval_quality.py
- Purpose: This test evaluates the quality of responses generated and checks for the presence of 'evolved_responses' in the returned response.

## test_eval_instruct.py
- Purpose: This test is created to check if the `eval_instruct` component works perfectly. The test ensures that the evolution process generates the expected results with proper keys and values in the response.

## test_llm_chat_generation.py
- Purpose: This test is created to verify if LLM generation for chats works perfectly.

## test_llm_text_generation.py
- Purpose: This test is created to validate if LLM generation on prompts works properly.

## test_magpie.py
- Purpose: This test is created to verify the ` MAGPIE` component. It ensures that a dataset is generated successfully and saved to a JSON file.

## test_ollama.py
- Purpose: This test is created  to verify the smooth working of Ollama provider.

## test_quality_scorer.py
- Purpose: This test is created to validate the `quality scorer` component. This test ensures that the quality scorer works perfectly by scoring the quality of responses based on given instructions and responses.

## test_sampling_params.py
- Purpose: This test is created to check if all the sampling parameters work perfectly.

## test_task_id_generator.py
- Purpose: This test is created to check if the `task_id_generator_function` passed externally works perfectly.

## test_various_limits_request_list_formation.py
- Purpose: This test is created to ensure that various limits on requests are applied and to verify if the `get_request_details` function creates appropriate requests.
