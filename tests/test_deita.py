#tests\test_deita.py
#The Test is created to check if the deita component works perfectly
from dataformer.components import Deita
def test_deita():
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

        assert results is not None
        for item in results:
            assert item is not None
            assert all(i in list(item.keys()) for i in ["evol_instruction_score", "evol_response_score", "embedding", "deita_score", "deita_score_computed_with", "nearest_neighbor_distance"]),"Response returned incorrect, missing some keys"
            assert item['evol_instruction_score'] is not None,"evol_instruction_score value is missing in Response"
            assert item['evol_response_score'] is not None,"evol_response_score value is missing in Response"
            assert item['embedding'] is not None,"embedding value is missing in Response"
            assert item['deita_score'] is not None,"deita_score value is missing in Response"
            assert item['deita_score_computed_with']is not None,"deita_score_computed_with value is missing in Response"
            assert item['nearest_neighbor_distance'] is not None,"nearest_neighbor_distance value is missing in Response"
    
