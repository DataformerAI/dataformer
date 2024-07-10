from dataformer.components import Deita

deita = Deita(
    max_rows=2, # number of rows desired after filtering
    diversity_threshold=0.9, # minimum cosine distance with respect to it's nearest neighbor, default value = 0.7
)

inputs = [
            {
                "evol_instruction_score": 0.3, # instruction score from complexity scorer
                "evol_response_score": 0.2, # response score from quality scorer
                "embedding": [-9.12727541, -4.24642847, -9.34933029],
            },
            {
                "evol_instruction_score": 0.6,
                "evol_response_score": 0.6,
                "embedding": [5.99395242, 0.7800955, 0.7778726],
            },
            {
                "evol_instruction_score": 0.7,
                "evol_response_score": 0.6,
                "embedding": [11.29087806, 10.33088036, 13.00557746],
            },
        ]

results = deita.filter(inputs)

print(results)