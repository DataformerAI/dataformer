# Deita

## Overview
The `Deita` component is designed to filter a list of inputs based on specified criteria, such as the maximum number of rows desired and a diversity threshold. It evaluates the inputs based on their instruction and response scores, as well as their embeddings.

## Example Usage

### Code Example
```python
from dataformer.components import Deita

# Initialize Deita with specific parameters
deita = Deita(
    max_rows=2,  # number of rows desired after filtering
    diversity_threshold=0.9,  # minimum cosine distance with respect to its nearest neighbor, default value = 0.7
)

# Define input data
inputs = [
    {
        "evol_instruction_score": 0.3,  # instruction score from complexity scorer
        "evol_response_score": 0.2,  # response score from quality scorer
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

# Filter the inputs using Deita
results = deita.filter(inputs)

# Print the results
for item in results:
    print(f"Evolved Instruction Score: {item['evol_instruction_score']}")
    print(f"Evolved Response Score: {item['evol_response_score']}")
    print(f"Embedding: {item['embedding']}")
    print(f"Deita Score: {item['deita_score']}")
    print(f"Deita Score Computed With: {item['deita_score_computed_with']}")
    print(f"Nearest Neighbor Distance: {item['nearest_neighbor_distance']}")
```

### Example Input
The input data consists of a list of dictionaries, each containing:
- `evol_instruction_score`: A score representing the complexity of the instruction.
- `evol_response_score`: A score representing the quality of the response.
- `embedding`: A list of numerical values representing the embedding of the input.

### Example Output
The output will be a filtered list of inputs based on the specified criteria. Each item in the output will include:
- `evol_instruction_score`: The evolved instruction score.
- `evol_response_score`: The evolved response score.
- `embedding`: The embedding of the input.
- `deita_score`: The score computed by Deita.
- `deita_score_computed_with`: The method used to compute the Deita score.
- `nearest_neighbor_distance`: The distance to the nearest neighbor.

### Sample Output
```
Evolved Instruction Score: 0.6
Evolved Response Score: 0.6
Embedding: [5.99395242, 0.7800955, 0.7778726]
Deita Score: 0.85
Deita Score Computed With: Method A
Nearest Neighbor Distance: 0.15
```