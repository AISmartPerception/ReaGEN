import io
import os
import random
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm, trange
import time
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from .mcts import MCTS, MCTSNode

from .utils_call import (
    generate_message,
    default_call,
    get_reward,
    reward_call,
    decompose_question_call,
    extract_final_answer,
    extract_final_answer_word,
    majority_vote,
    extract_sub_questions_and_answers,
    get_reward_answer,
    get_reward_subquestion,
    get_reward,
    get_reward_parallel,
    is_terminal_question,
    
)


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(
        self,
        question,
        sub_questions,
        sub_answers,
        depth,
        parent: "ReasoningMCTSNode" = None,
        r0=0.0,
        r1=0.0,
        cfg=None,
        model=None,
        image_url=None,
        tracker=None,
    ):
        # NOTE: do NOT call super().__init__(...), base MCTSNode has no __init__
        self._conf = None
        self.children = []
        self.depth = depth
        self._r0 = r0
        self._r1 = r1
        self._visited = False
        self.parent = parent
        self.tracker = tracker  # keep tracker

        self.question = question
        self.sub_questions = sub_questions
        self.sub_answers = sub_answers
        self.cfg = cfg
        self.model = model
        self.image_url = image_url

    def _child_node(self, question, sub_questions, sub_answers, r0, r1):
        return ReasoningMCTSNode(
            question,
            sub_questions,
            sub_answers,
            self.depth + 1,
            parent=self,
            r0=r0,
            r1=r1,
            cfg=self.cfg,
            model=self.model,
            image_url=self.image_url,
            tracker=self.tracker,  # pass same tracker to child
        )

    def gen_fn(self):
        if self.depth == self.cfg.prompt_method.decompose_question_num:
            # last sub question
            last_sub_question = True
        else:
            last_sub_question = False

        # pass tracker here
        candidate_paths = decompose_question_call(
            self.cfg,
            self.image_url,
            self.model,
            self.question,
            self.sub_questions,
            self.sub_answers,
            last_sub_question,
            tracker=self.tracker,
        )
        candidate_sub_questions = []
        candidate_sub_answers = []
        for j, path in enumerate(candidate_paths):
            sub_question, sub_answer = extract_sub_questions_and_answers(path, self.depth)
            if sub_question == "" or sub_answer == "":
                continue
            candidate_sub_questions.append(sub_question)
            candidate_sub_answers.append(sub_answer)

        # evaluate each candidate – also pass tracker
        candidate_rewards = get_reward_parallel(
            self.cfg,
            self.image_url,
            self.model,
            self.question,
            self.sub_questions,
            self.sub_answers,
            candidate_sub_questions,
            candidate_sub_answers,
            tracker=self.tracker,
        )
        return candidate_sub_questions, candidate_sub_answers, candidate_rewards

    def _get_children(self):
        self._visited = True
        if self.is_terminal:
            return self.children
        candidate_sub_questions, candidate_sub_answers, candidate_rewards = self.gen_fn()
        for i, (sub_question, sub_answer, reward) in enumerate(
            zip(candidate_sub_questions, candidate_sub_answers, candidate_rewards)
        ):
            sub_questions = self.sub_questions + [sub_question]
            sub_answers = self.sub_answers + [sub_answer]
            r0 = reward["sub_question_reward"]
            r1 = reward["sub_answer_reward"]
            self.children.append(
                self._child_node(self.question, sub_questions, sub_answers, r0, r1)
            )
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _static_terminal(self):
        if self.sub_answers:
            return is_terminal_question(self.question, self.sub_questions[-1])
        else:
            return False

    @property
    def is_terminal(self):
        return self._static_terminal()

    @property
    def reward(self):
        return math.sqrt(self._r0 * self._r1)  # [0, 1]

    def set_reward(self, r):
        self._r0 = r
        self._r1 = 1.0

    def __setstate__(self, state):
        # when unpickling
        self.__dict__.update(state)
        self.parent = None
        self.cfg = None
        self.model = None
        self.image_url = None

    def __getstate__(self):
        state = {
            "question": self.question,
            "sub_questions": self.sub_questions,
            "sub_answers": self.sub_answers,
            "depth": self.depth,
            "_r0": self._r0,
            "_r1": self._r1,
            "_visited": self._visited,
            "children": self.children,
        }
        return state

    def print(self):
        print("question: ", self.question)
        print("sub_questions: ", self.sub_questions)
        print("sub_answers: ", self.sub_answers)
        print("depth: ", self.depth)
        print("children: ", self.children)
        print("reward: ", self.reward)
        for child in self.children:
            print("child: ", child)
            child.print()


def mcts_forward_v8(query, img_path, model, cfg):

    tracker = {
        "student_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": [],
    }

    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_base64}"

    mcts = MCTS(w_exp=1, prior=True, aggr_reward="mean", aggr_child="max")
    root = ReasoningMCTSNode(
        query,
        [],
        [],
        1,
        cfg=cfg,
        model=model,
        image_url=image_url,
        tracker=tracker,
    )

    for _ in trange(cfg.prompt_method.mcts_rollouts):
        mcts.rollout(root, tracker=tracker)

    memory = {}

    max_n, max_r = mcts.max_mean_terminal(root)
    final_answer = max_n.sub_answers[-1]

    memory["max_mean_terminal"] = {}
    memory["max_mean_terminal"]["sub_questions"] = max_n.sub_questions
    memory["max_mean_terminal"]["sub_answers"] = max_n.sub_answers

    max_n, max_r = mcts.max_terminal(root)
    final_answer = max_n.sub_answers[-1]

    memory["max_terminal"] = {}
    memory["max_terminal"]["sub_questions"] = max_n.sub_questions
    memory["max_terminal"]["sub_answers"] = max_n.sub_answers

    terminal_nodes = mcts.max_vote_terminal(root)
    final_answers = [node[0].sub_answers[-1] for node in terminal_nodes]
    final_answers_word = [extract_final_answer_word(a) for a in final_answers]
    final_answer, index = majority_vote(final_answers_word)
    final_answer = final_answers[index]

    memory["max_vote_terminal"] = {}
    memory["max_vote_terminal"]["sub_questions"] = [node[0].sub_questions for node in terminal_nodes]
    memory["max_vote_terminal"]["sub_answers"] = [node[0].sub_answers for node in terminal_nodes]

    res = {
        "memory": memory,
        "response": final_answer,
        "root": root,
        "vrest_stats": tracker,  # final stats
    }

    return res