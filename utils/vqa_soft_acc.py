import json
import re
from typing import Iterable, List, Union

# --- Normalization helpers ----------------------------------------------------

# Remove common English articles
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", flags=re.IGNORECASE)
# Strip punctuation (you can swap to r"[^\w\s]" if you prefer)
_PUNCTUATION_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")


def normalize_text(text: str) -> str:
    """Lowercase, drop articles & punctuation, and collapse whitespace."""
    s = (text or "").lower().strip()
    s = _ARTICLE_RE.sub(" ", s)
    s = _PUNCTUATION_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_answer(pred: Union[str, dict]) -> str:
    """
    If pred is a JSON string/object with key 'answer', return that value.
    Otherwise return the string form of pred.
    """
    if isinstance(pred, dict):
        return str(pred.get("answer", ""))
    if isinstance(pred, str):
        try:
            obj = json.loads(pred)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"])
        except Exception:
            pass
        return pred
    return str(pred)


def vqa_soft_accuracy(pred: Union[str, dict], answers: Iterable[str]) -> float:
    """
    VQA-style soft accuracy on normalized strings:
        score = min(1.0, (# exact matches) / 3.0)
    `answers` can be a list of annotator answers or a single string.
    """
    # Normalize prediction (use only the answer text, not raw JSON)
    p = normalize_text(extract_answer(pred))

    # Collect references
    if answers is None:
        refs: List[str] = []
    elif isinstance(answers, (list, tuple)):
        refs = [a for a in answers if a is not None]
    else:
        refs = [answers]

    if not refs:
        return 0.0

    refs_norm = [normalize_text(a) for a in refs]
    matches = sum(1 for a in refs_norm if a == p)
    return min(1.0, matches / 3.0)


_PREFIX_RES = [
    re.compile(r"^answer\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"^final\s*answer\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"^assistant\s*[:\-]\s*", re.IGNORECASE),
]

def clean_generation(text: str) -> str:
    """Keep last non-empty line, strip common prefixes, and remove wrapping quotes/backticks."""
    if text is None:
        return ""
    s = text.strip()
    lines = [ln for ln in s.splitlines() if ln.strip()]
    s = lines[-1] if lines else s
    for rx in _PREFIX_RES:
        s = rx.sub("", s).strip()
    return s.strip("`'\" ").strip()

def keyword_soft_accuracy(pred: Union[str, dict], answers: Iterable[str]) -> float:
    """
    Whole-word containment fallback:
      - If strict soft accuracy is 0, award partial credit when normalized refs
        appear as whole words in the normalized prediction.
      - Score = min(1.0, hits / 3.0).
    """
    p = normalize_text(extract_answer(clean_generation(pred if isinstance(pred, str) else pred)))
    if not p or answers is None:
        return 0.0

    padded = f" {p} "
    refs = (answers if isinstance(answers, (list, tuple)) else [answers])
    hits = 0
    for a in refs:
        if not a:
            continue
        a_norm = normalize_text(a)
        if not a_norm:
            continue
        # whole-word search
        if re.search(rf"\b{re.escape(a_norm)}\b", padded):
            hits += 1
    return min(1.0, hits / 3.0)

def combined_accuracy(out: dict, gt) -> float:
    """
    Try VQA soft accuracy first; if it's zero, fall back to keyword soft accuracy.
    """
    # base = vqa_soft_accuracy(pred, answers)
    # return base if base > 0 else keyword_soft_accuracy(pred, answers)

    pred = out["rendered_answer"] if "rendered_answer" in out else out["direct_answer_rendered"]
    direct_pred = out["direct_answer_rendered"]

    # if isinstance(gt, tuple): 
    #     choise, str_answer = gt[0], gt[1]
    # else:
    #     choise, str_answer = gt, ""

    choise, str_answer = gt[0], gt[1]
    score, di_score = 0.0, 0.0

    if pred.lower() == choise.lower():
        score = 1.0
    elif len(pred) != 1:
        if pred.lower() in str_answer.lower() or str_answer.lower() in pred.lower():
            score = 1.0
 

    if direct_pred.lower() == choise.lower():
        di_score = 1.0
    elif len(direct_pred) != 1:
        if direct_pred.lower() in str_answer.lower() or str_answer.lower() in direct_pred.lower():
            di_score = 1.0
     
    return score, di_score


    # if pred.lower() == answer.lower():
    #     score = 1.0
    # else:
    #     score = 0.0
    
    # if direct_pred.lower() == answer.lower():
    #     di_score = 1.0
    # else:
    #     di_score = 0.0
    # return score, di_score

from difflib import SequenceMatcher
import numpy as np

def chain_similarity(chain1: List[str], chain2: List[str]) -> float:
    """
    Compute similarity between two stage sequences (list of stage names).
    """
    sm = SequenceMatcher(None, " ".join(chain1), " ".join(chain2))
    return sm.ratio()


def composite_reward(out: dict, answer: str, 
                    alpha: float = 0.1,
                    beta: float = 0.2, 
                    stages = None,
                    searched_chains = None) -> float:
    composite_score = 0.0
    # 1: Prediction Reward:
    score, direct_score = combined_accuracy(out, out['gt'])
    # stages = list(out['shared_blackboard'].keys())
    if stages is None:
        stages = out['stages']
    num_stages = len(stages)
    
    max_stages = 12
    
    composite_score += score
    if score > 0:
        # compute Stage Structural Reward:
        R_len = min(1.0, num_stages / max_stages) 
        composite_score -= alpha * R_len 
        
    elif score == 0:
        # compute Stage Structural Reward:
        if searched_chains:
            sims = [chain_similarity(stages, c) for c in searched_chains]
            mean_sim = np.mean(sims) if sims else 0.0
            composite_score -= beta * (mean_sim)  # higher diversity → higher reward
        else:
            composite_score -= beta * 0.05 

    
    return composite_score, score, direct_score

from typing import Any

def compute_token_cost(out: dict,processor: Any) -> int:
    token_cost = 0
    for stage_name, stage_output in out['stage_outputs'].items():
        if stage_name == 'DIRECT_ANSWER':
            continue
        stage_input_text_tokens_cost = stage_output['prompt_len']
        stage_input_image_token_cost = out['image_emb'][0].shape[1]

        text_output = stage_output['output'] if stage_name != 'ANSWER.CONSOLIDATION' else stage_output['raw_text']


        enc = processor.tokenizer(
            text_output,
            add_special_tokens=False,   # or True if you want BOS/EOS etc. counted
            return_tensors=None,
        )
        stage_output_text_tokens_cost = len(enc['input_ids'])

        token_cost += stage_input_text_tokens_cost + stage_input_image_token_cost + stage_output_text_tokens_cost



    return token_cost