from typing import Any, Dict, List, Mapping, Tuple
import re
from collections import Counter


_PRED_KEYS  = ("pred", "prediction", "answer", "y_hat", "output", "model_answer")
_REF_KEYS   = ("ref", "reference", "references", "gold", "label", "target",
               "answers", "possible_answers", "gt", "y")
_TRACE_KEYS = ("trace", "intermediate", "steps", "rationale", "chain_of_thought", "cots")
_TIME_KEYS  = ("latency", "time_s", "elapsed_s")
_TOKEN_KEYS = ("tokens", "token_usage", "usage")


def _flatten_fb(fb) -> List[Mapping[str, Any]]:
    if isinstance(fb, dict):
        for k in ("records", "items", "examples", "traces", "data"):
            if k in fb and isinstance(fb[k], list):
                return fb[k]
        return [fb]
    if isinstance(fb, (list, tuple)):
        return list(fb)
    return []

def _pick_first(d: Mapping[str, Any], keys: Tuple[str, ...], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _normalize(txt: str) -> str:
    s = txt.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s*([,.:;!?()\[\]])\s*", r"\1", s)
    return s

def _to_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+|[%$€¥]+|==|!=|<=|>=|[-+*/^()=,.:;]", s)

def _f1(pred: str, refs: List[str]) -> float:
    pt = Counter(_tokenize(_normalize(pred)))
    best = 0.0
    for r in refs:
        rt = Counter(_tokenize(_normalize(r)))
        common = sum((pt & rt).values())
        if common == 0:
            continue
        prec = common / max(1, sum(pt.values()))
        rec  = common / max(1, sum(rt.values()))
        f1 = 2 * prec * rec / max(1e-12, (prec + rec))
        best = max(best, f1)
    return best

def _exact_match(pred: str, refs: List[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(r) for r in refs)

def _unit_mismatch(pred: str, refs: List[str]) -> bool:
    def nums_units(s):
        nums  = re.findall(r"\b\d+(?:\.\d+)?\b", s)
        units = re.findall(r"\b([a-zA-Zμ%]+)\b", s)
        return set(nums), set(u for u in units if not re.match(r"^(and|or|the|to|a|an|is|are)$", u))
    pn, pu = nums_units(pred)
    for r in refs:
        rn, ru = nums_units(r)
        if pn and rn and (pu != ru):
            return True
    return False

def _format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"
