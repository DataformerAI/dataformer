import math
from typing import Any, Dict, List
import re
from dataformer.llms import AsyncLLM


class MCTS:
    def __init__(
        self,
        llm: AsyncLLM,
        max_iter: int = 16,
        C: float = 1.4,
    ):
        self.llm = llm
        self.max_iter = max_iter
        self.C = C

    def generate(self, requests: List[str]) -> List[Dict[str, Any]]:
        responses = []
        for request in requests:
            response = self.mcts_loop(request)
            responses.append(response)
        return responses

    def mcts_loop(self, query: str) -> Dict[str, Any]:
        to_explore = []
        to_explore_reward = {}
        history_bank = {}
        hints_bank = {}
        ucb_bank = {}
        fathers = {}
        childs = {}
        answers_list = []

        # Get initial weak answer
        weak_answer, history = self.get_weak_answer(query)
        history_bank[weak_answer] = tuple(history)
        answers_list = [weak_answer]
        to_explore = [weak_answer]
        childs[weak_answer] = []
        fathers[weak_answer] = None
        self.sampling_reward(query, weak_answer, to_explore_reward)

        for _ in range(self.max_iter):
            filtered_to_explore = self.filter_mature_node(childs, to_explore, to_explore_reward)
            weak_answer = self.get_best_explore_from_ucb(filtered_to_explore, ucb_bank)
            self.sampling_reward(query, weak_answer, to_explore_reward)

            hints, answer, history = self.step(query, weak_answer, history=history_bank[weak_answer])
            self.add_to_hints_bank(hints, weak_answer, hints_bank)
            history_bank[answer] = tuple(history)
            to_explore.append(answer)
            self.sampling_reward(query, answer, to_explore_reward)
            fathers[answer] = weak_answer
            childs[answer] = []
            self.add_to_childs(weak_answer, answer, childs)
            answers_list.append(answer)

            self.update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank)

        best_answer = max(answers_list, key=lambda x: max(to_explore_reward.get(x, [-float('inf')])))
        return {
            'query': query,
            'answers_list': answers_list,
            'best_answer': best_answer,
        }

    def get_weak_answer(self, query: str) -> tuple:
        prompt = f"Question: {query}\nThe response should begin with [reasoning process]...[Verification]... and end with ####\nLet's think step by step."
        response = self.llm.generate([{'messages': [{'role': 'user', 'content': prompt}]}], use_cache=False)
        return response[0][1]['choices'][0]['message']['content'], [prompt, response[0][1]['choices'][0]['message']['content']]

    def step(self, query: str, weak_answer: str, history: List[str]) -> tuple:
        hints_prompt = f"Question: {query}\nSince we have a weak Answer, could you provide me with a reflection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score!\nLet's think step by step."
        hints = self.llm.generate([{'messages': [{'role': 'user', 'content': msg} for msg in history] + [{'role': 'user', 'content': hints_prompt}]}], use_cache=False)
        hints = hints[0][1]['choices'][0]['message']['content']
        new_history = list(history) + [hints_prompt, hints]

        answer_prompt = f"Question: {query}\nPlease refine your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with ####\nLet's think step by step."
        answer = self.llm.generate([{'messages': [{'role': 'user', 'content': msg} for msg in new_history] + [{'role': 'user', 'content': answer_prompt}]}], use_cache=False)
        answer = answer[0][1]['choices'][0]['message']['content']
        final_history = list(new_history) + [answer_prompt, answer]

        return hints, answer, final_history

    def sampling_reward(self, query: str, answer: str, to_explore_reward: Dict[str, List[float]]):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward_prompt = f"Question: {query}\nAnswer:{answer}\nAnalyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [-100,+100], e.g. from -100 to +100. \nResponse format:\n[Analyst]...[Score]..."
        reward_response = self.llm.generate([{'messages': [{'role': 'user', 'content': reward_prompt}]}], use_cache=False)
        reward_text = reward_response[0][1]['choices'][0]['message']['content']
        scores = re.findall(r'-?\d+', reward_text.split('Score')[-1])
        reward = float(scores[-1]) if scores else 0
        if reward >= 95:
            reward = 50
        to_explore_reward[answer].append(reward)

    @staticmethod
    def add_to_hints_bank(hints: str, weak_answer: str, hints_bank: Dict[str, List[str]]):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    @staticmethod
    def add_to_childs(father: str, child: str, childs: Dict[str, List[str]]):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    @staticmethod
    def filter_mature_node(childs: Dict[str, List[str]], to_explore: List[str], to_explore_reward: Dict[str, List[float]], max_expand: int = 3):
        filtered_to_explore = []
        avg_reward = {node: (min(to_explore_reward[node]) + sum(to_explore_reward[node]) / len(to_explore_reward[node])) / 2 for node in to_explore}

        for node in to_explore:
            if len(childs.get(node, [])) < max_expand or max([avg_reward.get(child, -float('inf')) for child in childs.get(node, [])]) < avg_reward.get(node, -float('inf')):
                filtered_to_explore.append(node)

        return filtered_to_explore

    def get_best_explore_from_ucb(self, to_explore: List[str], ucb_bank: Dict[str, float]):
        return max(to_explore, key=lambda node: ucb_bank.get(node, float('-inf')))

    def update_ucb(self, fathers: Dict[str, str], childs: Dict[str, List[str]], to_explore: List[str], to_explore_reward: Dict[str, List[float]], ucb_bank: Dict[str, float]):
        visit_count = {node: len(to_explore_reward.get(node, [])) for node in to_explore}
        avg_reward = {node: (min(to_explore_reward.get(node, [0])) + sum(to_explore_reward.get(node, [0])) / len(to_explore_reward.get(node, [1]))) / 2 for node in to_explore}

        leaves = set(to_explore) - set(fathers.values())

        for leaf in leaves:
            father_rewards = to_explore_reward.get(fathers.get(leaf), [])
            leaf_rewards = to_explore_reward.get(leaf, [])
            ucb_bank[leaf] = self.compute_ucb(avg_reward[leaf], len(father_rewards), len(leaf_rewards))

        nodes_to_update = list(leaves)
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = fathers.get(node)
                if father is not None:
                    if father not in ucb_bank:
                        new_nodes_to_update.add(father)
                    if father in ucb_bank:
                        child_reward = [avg_reward[child] for child in childs[father]]
                        father_reward = (avg_reward[father] + max(child_reward)) / 2
                        father_rewards = to_explore_reward.get(fathers.get(father), [])
                        father_leaf_rewards = to_explore_reward.get(father, [])
                        ucb_bank[father] = self.compute_ucb(father_reward, len(father_rewards), len(father_leaf_rewards))
            nodes_to_update = list(new_nodes_to_update)

    def compute_ucb(self, r_c: float, N_n: int, N_c: int) -> float:
        return r_c + self.C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))