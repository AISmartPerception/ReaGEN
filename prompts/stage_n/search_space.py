import json

class SearchNode:
    """Represents one stage node in the explored search tree."""
    def __init__(self, name=None):
        self.name = name
        self.stage_result = None  # stores the stage output dict
        self.next = {}            # child stages: name -> SearchNode
        self.final_output = None
        self.final = False
        self.reward = None
        self.score = None

    def get_or_create(self, stage_name: str):
        """Return the child node for stage_name, creating it if missing."""
        if stage_name not in self.next:
            self.next[stage_name] = SearchNode(stage_name)
        return self.next[stage_name]

    def get(self, stage_name: str):
        """Return the child node if exists, else None."""
        return self.next.get(stage_name, None)

    def to_dict(self):
        """Convert recursively to a JSON-serializable dictionary."""
        return_dict = {
            "name": self.name,
            "stage_result": self.stage_result['output'] if self.stage_result is not None else None,
            "next": {k: v.to_dict() for k, v in self.next.items()},
        }
        if self.final:
            return_dict["final_output"] = self.final_output
            return_dict["reward"] = self.reward
            return_dict["score"] = self.score
            
        return return_dict

    def clear_final(self):
        """Remove only final info from this node."""
        # self.final_output = None
        # self.reward = None
        # self.score = None
        self.final = False

    def clear_all_finals(self):
        """Recursively remove final info from this subtree."""
        self.clear_final()
        for child in self.next.values():
            child.clear_all_finals()
            
    # def _safe_stage_result(self):
    #     """Clean the stage_result for JSON dumping (avoid ndarray errors)."""
    #     if self.stage_result is None:
    #         return None
    #     cleaned = {}
    #     for k, v in self.stage_result.items():
    #         if isinstance(v, (str, int, float, bool, type(None))):
    #             cleaned[k] = v
    #         else:
    #             cleaned[k] = str(type(v))  # fallback for non-serializable items
    #     return cleaned


class SearchSpace:
    """Stores explored multi-stage reasoning chains as a tree of SearchNodes."""
    def __init__(self):
        self.root = SearchNode("ROOT")

    def insert(self, stage_seq: list[str], stage_result: dict):
        """Insert a new stage result following the stage sequence path."""
        node = self.root
        for s in stage_seq:
            node = node.get_or_create(s)
        node.stage_result = stage_result

    def add_reward(self, stage_seq: list[str], reward: float, score: float, final_output: dict):
        """Add a reward to the final node of the stage sequence."""
        node = self.root
        for s in stage_seq:
            node = node.get(s)

        if node.reward is not None and node.reward < 0:
            node.reward = node.reward + reward
        else:
            node.reward = reward
        node.final_output = final_output
        node.score = score
        node.final = True
        
    def get_cached(self, stage_seq: list[str]):
        """Return cached stage result for a given sequence if exists."""
        node = self.root
        for s in stage_seq:
            node = node.get(s)
            if node is None:
                return None
        return node.stage_result

    def get_node(self, stage_seq: list[str]):
        """Return node at path if exists."""
        node = self.root
        for s in stage_seq:
            node = node.get(s)
            if node is None:
                return None
        return node

    def to_dict(self):
        """Serialize entire tree to JSON-compatible dict."""
        return self.root.to_dict()


    def clear_final_at(self, stage_seq: list[str]) -> bool:
        """
        Clear only the final info at the node for stage_seq.
        Returns True if cleared, False if the path doesn't exist.
        """
        node = self.get_node(stage_seq)
        if node is None:
            return False
        node.clear_final()
        return True

    def clear_all_finals(self):
        """Clear final info from all nodes in the search space."""
        self.root.clear_all_finals()
    
    
def flatten_search_space(node: SearchNode, prefix=None, stage_outputs=None):
    """
    Recursively flatten the search tree into a list of *complete* reasoning chains
    (i.e., only those ending with node.final == True).
    
    Each returned chain includes:
      - stage_seq: list of stage names
      - stage_outputs: list of per-stage outputs (text)
      - final_output: dict from final stage
      - reward: float
    """
    if prefix is None:
        prefix = []
    if stage_outputs is None:
        stage_outputs = []

    chains = []

    # skip root (which has no actual stage_result)
    if node.name != "ROOT" and node.stage_result is not None:
        stage_outputs = stage_outputs + [node.stage_result.get("output", None)]
        prefix = prefix + [node.name]

    # if this is a final node, record the full chain
    if node.final:
        chains.append({
            "stage_seq": prefix,
            "stage_outputs": stage_outputs,
            "final_output": node.final_output,
            "reward": node.reward,
            "score": node.score,
        })

    # recursively explore next stages
    for next_name, next_node in node.next.items():
        chains.extend(flatten_search_space(next_node, prefix, stage_outputs))

    return chains