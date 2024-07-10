from typing import List, Literal, Dict, Any
import numpy as np

class Deita:
    def __init__(self, max_rows: int = 1, diversity_threshold: float = 0.7, normalize_embeddings: bool = True, distance_metric: Literal["cosine", "manhattan"] = "cosine"):
        self.max_rows = max_rows
        self.diversity_threshold = diversity_threshold
        self.normalize_embeddings = normalize_embeddings
        self.distance_metric = distance_metric

    def compute_deita_score(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for input_ in inputs:
            evol_instruction_score = input_.get("evol_instruction_score")
            evol_response_score = input_.get("evol_response_score")

            if evol_instruction_score and evol_response_score:
                deita_score = evol_instruction_score * evol_response_score
                score_computed_with = ["evol_instruction_score", "evol_response_score"]
            elif evol_instruction_score:
                deita_score = evol_instruction_score
                score_computed_with = ["evol_instruction_score"]
            elif evol_response_score:
                deita_score = evol_response_score
                score_computed_with = ["evol_response_score"]
            else:
                deita_score = 0
                score_computed_with = []

            input_.update(
                {
                    "deita_score": deita_score,
                    "deita_score_computed_with": score_computed_with,
                }
            )
        return inputs


    def compute_nearest_neighbor(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Computes the cosine distance between the embeddings of the instruction-response
        pairs and the nearest neighbor.
        """
        embeddings = np.array([input["embedding"] for input in inputs])
        if self.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)

        if self.distance_metric == "cosine":
            distances = self.cosine_distance(embeddings)
        else:
            distances = self.manhattan_distance(embeddings)

        for distance, input in zip(distances, inputs):
            input["nearest_neighbor_distance"] = round(float(distance), 3)
        return inputs

    def _normalize_embeddings(self, embeddings:np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def cosine_distance(self, embeddings:np.array) -> np.array:
        cosine_similarity = np.dot(embeddings, embeddings.T)
        cosine_distance = 1 - cosine_similarity
        np.fill_diagonal(cosine_distance, np.inf)
        return np.min(cosine_distance, axis=1)

    def manhattan_distance(self, embeddings:np.array) -> np.array:
        manhattan_distance = np.abs(embeddings[:, None] - embeddings).sum(-1)
        np.fill_diagonal(manhattan_distance, np.inf)
        return np.min(manhattan_distance, axis=1)

    def filter(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inputs = self.compute_deita_score(inputs)
        inputs = self.compute_nearest_neighbor(inputs)
        inputs.sort(key=lambda x: (x["deita_score"], x['nearest_neighbor_distance']), reverse=True)
        
        selected_rows = []
        for input in inputs:
            if len(selected_rows) >= self.max_rows:
                break
            if input["nearest_neighbor_distance"] >= self.diversity_threshold:
                selected_rows.append(input)
        return selected_rows